"""Empirical label-horizon tuning on the engineered 15m feature spine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.profile_manager import LabelProfileManager
from quant_system.label_generation.utils import (
    compute_bos_cont_labels,
    compute_edp_labels,
    compute_eop_labels,
    compute_flow_1h_labels,
    compute_hazard_labels,
    compute_liq_flow_labels,
    compute_momo_labels,
)
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_num, get_logger

LOG = get_logger("label_tuner")


LABEL_FUNCTIONS: Dict[str, Callable[[pd.DataFrame, Dict[str, Any]], Any]] = {
    "liq_flow": compute_liq_flow_labels,
    "bos_cont": compute_bos_cont_labels,
    "momo": compute_momo_labels,
    "flow_1h": compute_flow_1h_labels,
    "eop": compute_eop_labels,
    "edp": compute_edp_labels,
    "hazard": compute_hazard_labels,
}


DEFAULT_SWEEPS: Dict[str, List[Dict[str, Any]]] = {
    "liq_flow": [
        {"horizon_bars": 8},
        {"horizon_bars": 12},
        {"horizon_bars": 16},
        {"horizon_bars": 24},
    ],
    "bos_cont": [
        {"horizon_bars": 24},
        {"horizon_bars": 36},
        {"horizon_bars": 48},
        {"horizon_bars": 64},
        {"horizon_bars": 72},
    ],
    "momo": [
        {"min_horizon": 4, "max_horizon": 6},
        {"min_horizon": 4, "max_horizon": 8},
        {"min_horizon": 6, "max_horizon": 10},
        {"min_horizon": 8, "max_horizon": 12},
    ],
    "flow_1h": [
        {"min_horizon": 4, "max_horizon": 8},
        {"min_horizon": 4, "max_horizon": 12},
        {"min_horizon": 6, "max_horizon": 12},
        {"min_horizon": 6, "max_horizon": 16},
    ],
    "eop": [
        {"horizon_bars": 48},
        {"horizon_bars": 72},
        {"horizon_bars": 96},
        {"horizon_bars": 120},
        {"horizon_bars": 144},
    ],
    "edp": [
        {"horizon_bars": 48},
        {"horizon_bars": 72},
        {"horizon_bars": 96},
        {"horizon_bars": 120},
        {"horizon_bars": 144},
    ],
    "hazard": [
        {"horizon_bars": 24},
        {"horizon_bars": 36},
        {"horizon_bars": 48},
        {"horizon_bars": 64},
        {"horizon_bars": 72},
    ],
}


class LabelEmpiricalTuner:
    EXCLUDE_COLS = {
        "dt",
        "timestamp",
        "label_liq_flow",
        "label_bos_cont",
        "label_momo",
        "label_flow_1h",
        "label_eop",
        "label_edp",
        "hazard_event",
        "hazard_time",
    }

    def __init__(self, config_loader: ConfigLoader):
        self.cfg_loader = config_loader
        self.labels_cfg = deepcopy(config_loader.load_yaml("labels.yaml")["labels"])
        self.profile_manager = LabelProfileManager()

    def tune(
        self,
        df15: pd.DataFrame,
        *,
        tasks: Iterable[str] | None = None,
        output_dir: str | None = None,
        auto_promote: bool = True,
        min_improvement: float = 0.01,
        resume: bool = True,
    ) -> Dict[str, Any]:
        selected_tasks = list(tasks or DEFAULT_SWEEPS.keys())
        dataset_signature = self._dataset_signature(df15)
        console_rule("Label Horizon Tuning", style="magenta")
        console_kv(
            "Tuning Plan",
            {
                "rows": fmt_num(len(df15)),
                "tasks": ", ".join(selected_tasks),
                "output_dir": output_dir or "-",
                "auto_promote": auto_promote,
                "min_improvement": min_improvement,
                "resume": bool(resume),
                "dataset_signature": dataset_signature,
            },
            style="magenta",
        )

        out_root: Optional[Path] = None
        progress_path: Optional[Path] = None
        progress_events_path: Optional[Path] = None
        progress_state: Dict[str, Any] = {}
        started_ts = self._utc_now_iso()
        started_wall = datetime.now(timezone.utc)
        candidate_totals: Dict[str, int] = {}
        candidate_names_by_task: Dict[str, set[str]] = {}
        for task in selected_tasks:
            if task not in LABEL_FUNCTIONS:
                raise ValueError(f"Unknown label tuning task: {task}")
            base_cfg = deepcopy(self.labels_cfg[task])
            candidates = self._build_candidates(task, base_cfg)
            candidate_totals[task] = len(candidates)
            candidate_names_by_task[task] = {
                self._candidate_name(task, patch)
                for patch in candidates
            }
        total_candidates = int(sum(candidate_totals.values()))
        console_kv(
            "Task Candidate Counts",
            {**{task: candidate_totals[task] for task in selected_tasks}, "total_candidates": total_candidates},
            style="cyan",
        )

        existing_completed_by_task = {task: 0 for task in selected_tasks}
        if output_dir:
            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            progress_path = out_root / "progress.json"
            progress_events_path = out_root / "progress.ndjson"

            if resume:
                self._seed_task_checkpoints_from_progress_events(
                    out_root,
                    selected_tasks,
                    dataset_signature=dataset_signature,
                )
                for task in selected_tasks:
                    existing_df = self._load_task_checkpoint_frame(
                        out_root,
                        task,
                        dataset_signature=dataset_signature,
                    )
                    if existing_df.empty or "candidate_name" not in existing_df.columns:
                        continue
                    existing_names = set(existing_df["candidate_name"].dropna().astype(str).tolist())
                    done_names = existing_names & candidate_names_by_task[task]
                    existing_completed_by_task[task] = int(len(done_names))

            resumed_candidates = int(sum(existing_completed_by_task.values()))
            resumed_tasks = int(
                sum(
                    1
                    for task in selected_tasks
                    if existing_completed_by_task[task] >= int(candidate_totals[task])
                )
            )
            first_pending = next(
                (
                    task
                    for task in selected_tasks
                    if existing_completed_by_task[task] < int(candidate_totals[task])
                ),
                None,
            )
            pct_complete = float(resumed_candidates / max(total_candidates, 1))
            pct_remaining = float(1.0 - pct_complete)

            progress_state = {
                "status": "running",
                "started_at": started_ts,
                "updated_at": started_ts,
                "rows": int(len(df15)),
                "dataset_signature": dataset_signature,
                "tasks_total": int(len(selected_tasks)),
                "tasks_completed": resumed_tasks,
                "task_order": selected_tasks,
                "current_task": first_pending,
                "candidates_total": total_candidates,
                "candidates_completed": resumed_candidates,
                "candidates_remaining": int(max(total_candidates - resumed_candidates, 0)),
                "eta_seconds": None,
                "eta_hms": None,
                "elapsed_seconds": 0,
                "elapsed_hms": "00:00",
                "percent_complete": pct_complete,
                "percent_remaining": pct_remaining,
                "tasks": {
                    task: {
                        "status": "done" if existing_completed_by_task[task] >= int(candidate_totals[task]) else "pending",
                        "candidates_total": int(candidate_totals[task]),
                        "candidates_completed": int(existing_completed_by_task[task]),
                    }
                    for task in selected_tasks
                },
            }
            if resume and resumed_candidates > 0:
                console_stage(
                    "Resume checkpoint",
                    (
                        f"reusing {resumed_candidates}/{total_candidates} scored candidates | "
                        f"tasks_done={resumed_tasks}/{len(selected_tasks)}"
                    ),
                    status="info",
                )
            self._write_progress_snapshot(progress_path, progress_state)
            self._append_progress_event(
                progress_events_path,
                {
                    "event": "tuning_started",
                    "timestamp": started_ts,
                    "task_order": selected_tasks,
                    "resume": bool(resume),
                    "resumed_candidates": progress_state["candidates_completed"],
                    "total_candidates": total_candidates,
                    "dataset_signature": dataset_signature,
                },
            )

        results: Dict[str, Any] = {}
        promotions: Dict[str, Dict[str, Any]] = {}
        try:
            for task_idx, task in enumerate(selected_tasks, start=1):
                if task not in LABEL_FUNCTIONS:
                    raise ValueError(f"Unknown label tuning task: {task}")
                task_checkpoint_path = self._task_partial_path(out_root, task) if out_root is not None else None
                task_final_path = self._task_final_path(out_root, task) if out_root is not None else None
                task_resume_path = task_checkpoint_path
                if (
                    resume
                    and task_resume_path is not None
                    and (not task_resume_path.exists())
                    and task_final_path is not None
                    and task_final_path.exists()
                ):
                    task_resume_path = task_final_path
                task_already_scored = int(existing_completed_by_task.get(task, 0))
                task_was_done = bool(
                    progress_state.get("tasks", {}).get(task, {}).get("status") == "done"
                )
                console_rule(f"Task {task_idx}/{len(selected_tasks)} | {task}", style="cyan")
                console_stage(
                    "Task started",
                    f"candidates={candidate_totals[task]} already_scored={task_already_scored}",
                    status="info",
                )

                if progress_path is not None:
                    progress_state["current_task"] = None if task_was_done else task
                    progress_state["updated_at"] = self._utc_now_iso()
                    if not task_was_done:
                        progress_state["tasks"][task]["status"] = "running"
                    self._write_progress_snapshot(progress_path, progress_state)

                def _progress_hook(evt: Dict[str, Any]) -> None:
                    if progress_path is None:
                        return
                    progress_state["updated_at"] = self._utc_now_iso()
                    progress_state["candidates_completed"] = int(progress_state["candidates_completed"]) + 1
                    tstate = progress_state["tasks"][task]
                    tstate["candidates_completed"] = max(
                        int(tstate.get("candidates_completed", 0)),
                        int(evt.get("candidate_index", tstate.get("candidates_completed", 0))),
                    )
                    tstate["candidate_name"] = evt.get("candidate_name")
                    tstate["last_objective_score"] = evt.get("objective_score")
                    tstate["last_cv_ap"] = evt.get("cv_ap")
                    tstate["last_cv_auc"] = evt.get("cv_auc")
                    tstate["last_positive_rate"] = evt.get("positive_rate")

                    global_done = int(
                        sum(
                            int(v.get("candidates_completed", 0))
                            for v in progress_state.get("tasks", {}).values()
                        )
                    )
                    progress_state["candidates_completed"] = global_done
                    done = float(global_done)
                    total = max(float(progress_state["candidates_total"]), 1.0)
                    elapsed = max((datetime.now(timezone.utc) - started_wall).total_seconds(), 1e-6)
                    rate = done / elapsed
                    remaining = max(total - done, 0.0)
                    eta = int(remaining / rate) if rate > 0 else None
                    progress_state["eta_seconds"] = eta
                    progress_state["eta_hms"] = self._human_duration(eta)
                    progress_state["elapsed_seconds"] = int(elapsed)
                    progress_state["elapsed_hms"] = self._human_duration(elapsed)
                    progress_state["percent_complete"] = float(done / total)
                    progress_state["percent_remaining"] = float(1.0 - progress_state["percent_complete"])
                    progress_state["candidates_remaining"] = int(max(progress_state["candidates_total"] - progress_state["candidates_completed"], 0))

                    task_done = int(evt.get("candidate_index", 0))
                    task_total = max(int(evt.get("candidate_total", 0)), 1)
                    task_pct = (100.0 * task_done) / task_total
                    global_pct = 100.0 * progress_state["percent_complete"]
                    remaining_pct = 100.0 * progress_state["percent_remaining"]
                    console_stage(
                        "Progress",
                        (
                            f"{task} {task_done}/{task_total} ({task_pct:.1f}%) | "
                            f"global {progress_state['candidates_completed']}/{progress_state['candidates_total']} "
                            f"({global_pct:.1f}% done, {remaining_pct:.1f}% left) | "
                            f"elapsed={progress_state['elapsed_hms']} eta={progress_state['eta_hms'] or '-'} | "
                            f"objective={float(evt.get('objective_score', 0.0)):.4f}"
                        ),
                        status="info",
                    )

                    self._write_progress_snapshot(progress_path, progress_state)
                    self._append_progress_event(
                        progress_events_path,
                        {
                            "event": "candidate_scored",
                            "timestamp": progress_state["updated_at"],
                            "task": task,
                            **evt,
                            "dataset_signature": dataset_signature,
                            "global_candidates_completed": progress_state["candidates_completed"],
                            "global_candidates_total": progress_state["candidates_total"],
                            "eta_seconds": eta,
                        },
                    )

                result_df, recommended = self._tune_task(
                    df15,
                    task,
                    progress_hook=_progress_hook,
                    checkpoint_path=task_checkpoint_path,
                    resume_path=task_resume_path,
                    resume=resume,
                    dataset_signature=dataset_signature,
                )
                if result_df.empty:
                    raise RuntimeError(f"Label tuning produced no candidate rows for task={task}")
                baseline_rows = result_df[result_df["is_baseline"] == 1]
                baseline = (
                    baseline_rows.iloc[0].to_dict()
                    if not baseline_rows.empty
                    else result_df.iloc[0].to_dict()
                )
                improvement = float(recommended["objective_score"]) - float(baseline["objective_score"])
                should_promote = (
                    bool(auto_promote)
                    and bool(recommended["is_baseline"] == 0)
                    and improvement >= float(min_improvement)
                )
                if task_final_path is not None:
                    task_final_path.parent.mkdir(parents=True, exist_ok=True)
                    result_df.to_csv(task_final_path, index=False)
                if task_checkpoint_path is not None and task_checkpoint_path.exists():
                    task_checkpoint_path.unlink()
                existing_completed_by_task[task] = int(candidate_totals[task])
                results[task] = {
                    "results": result_df.to_dict(orient="records"),
                    "recommended": recommended,
                    "baseline": baseline,
                    "improvement_vs_baseline": improvement,
                    "promoted": should_promote,
                }
                console_stage(
                    f"{task} tuned",
                    (
                        f"recommended={recommended['candidate_name']} "
                        f"objective={recommended['objective_score']:.4f} "
                        f"baseline={baseline['objective_score']:.4f} "
                        f"delta={improvement:.4f}"
                    ),
                    status="ok",
                )
                if should_promote:
                    promotions[task] = {
                        key: recommended[key]
                        for key in ("horizon_bars", "min_horizon", "max_horizon")
                        if key in recommended and pd.notna(recommended[key])
                    }

                if progress_path is not None:
                    progress_state["current_task"] = None
                    progress_state["updated_at"] = self._utc_now_iso()
                    progress_state["tasks"][task]["status"] = "done"
                    progress_state["tasks"][task]["candidates_completed"] = int(candidate_totals[task])
                    progress_state["tasks"][task]["recommended"] = recommended.get("candidate_name")
                    progress_state["tasks"][task]["improvement_vs_baseline"] = improvement
                    progress_state["tasks"][task]["promoted"] = bool(should_promote)
                    progress_state["tasks_completed"] = int(
                        sum(
                            1
                            for state in progress_state.get("tasks", {}).values()
                            if str(state.get("status")) == "done"
                        )
                    )
                    global_done = int(
                        sum(
                            int(v.get("candidates_completed", 0))
                            for v in progress_state.get("tasks", {}).values()
                        )
                    )
                    progress_state["candidates_completed"] = global_done
                    total = max(float(progress_state["candidates_total"]), 1.0)
                    progress_state["percent_complete"] = float(global_done / total)
                    progress_state["percent_remaining"] = float(1.0 - progress_state["percent_complete"])
                    progress_state["candidates_remaining"] = int(max(total_candidates - global_done, 0))
                    elapsed = max((datetime.now(timezone.utc) - started_wall).total_seconds(), 1e-6)
                    progress_state["elapsed_seconds"] = int(elapsed)
                    progress_state["elapsed_hms"] = self._human_duration(elapsed)
                    self._write_progress_snapshot(progress_path, progress_state)
                    self._append_progress_event(
                        progress_events_path,
                        {
                            "event": "task_completed",
                            "timestamp": progress_state["updated_at"],
                            "task": task,
                            "recommended": recommended.get("candidate_name"),
                            "improvement_vs_baseline": improvement,
                            "promoted": bool(should_promote),
                            "resumed_task": bool(task_was_done),
                            "dataset_signature": dataset_signature,
                        },
                    )
                tasks_left = (
                    len(selected_tasks) - int(progress_state.get("tasks_completed", task_idx))
                    if progress_state
                    else len(selected_tasks) - task_idx
                )
                candidates_left = max(total_candidates - int(progress_state.get("candidates_completed", 0)), 0) if progress_state else 0
                console_stage(
                    "Remaining",
                    f"tasks_left={tasks_left} candidates_left={candidates_left}",
                    status="info",
                )
        except Exception as exc:
            if progress_path is not None:
                progress_state["status"] = "failed"
                progress_state["updated_at"] = self._utc_now_iso()
                progress_state["finished_at"] = progress_state["updated_at"]
                progress_state["error"] = str(exc)
                self._write_progress_snapshot(progress_path, progress_state)
                self._append_progress_event(
                    progress_events_path,
                    {
                        "event": "tuning_failed",
                        "timestamp": progress_state["updated_at"],
                        "error": str(exc),
                        "dataset_signature": dataset_signature,
                    },
                )
            raise

        if output_dir:
            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            for task, payload in results.items():
                pd.DataFrame(payload["results"]).to_csv(out_root / f"{task}_tuning.csv", index=False)
            manifest = {
                task: {
                    "recommended": payload["recommended"],
                    "baseline": payload["baseline"],
                    "improvement_vs_baseline": payload["improvement_vs_baseline"],
                    "promoted": payload["promoted"],
                }
                for task, payload in results.items()
            }
            (out_root / "recommended_label_horizons.json").write_text(
                json.dumps(manifest, indent=2, default=self._json_default),
                encoding="utf-8",
            )

        if promotions:
            self.profile_manager.promote(
                tasks=promotions,
                source_summary={
                    "auto_promote": auto_promote,
                    "min_improvement": min_improvement,
                    "tasks": {
                        task: {
                            "candidate_name": results[task]["recommended"]["candidate_name"],
                            "objective_score": results[task]["recommended"]["objective_score"],
                            "baseline_score": results[task]["baseline"]["objective_score"],
                            "improvement_vs_baseline": results[task]["improvement_vs_baseline"],
                        }
                        for task in promotions
                    },
                },
            )
        else:
            console_stage(
                "Label profile unchanged",
                "no challenger cleared the promotion threshold; defaults/current active profile stay in force",
                status="info",
            )

        if progress_path is not None:
            progress_state["status"] = "completed"
            progress_state["updated_at"] = self._utc_now_iso()
            progress_state["finished_at"] = progress_state["updated_at"]
            progress_state["current_task"] = None
            progress_state["eta_seconds"] = 0
            progress_state["eta_hms"] = "00:00"
            progress_state["percent_complete"] = 1.0
            progress_state["percent_remaining"] = 0.0
            progress_state["candidates_remaining"] = 0
            elapsed_all = max((datetime.now(timezone.utc) - started_wall).total_seconds(), 0.0)
            progress_state["elapsed_seconds"] = int(elapsed_all)
            progress_state["elapsed_hms"] = self._human_duration(elapsed_all)
            self._write_progress_snapshot(progress_path, progress_state)
            self._append_progress_event(
                progress_events_path,
                {
                    "event": "tuning_completed",
                    "timestamp": progress_state["updated_at"],
                    "tasks_completed": progress_state.get("tasks_completed"),
                    "candidates_completed": progress_state.get("candidates_completed"),
                    "dataset_signature": dataset_signature,
                },
            )

        return results

    def _tune_task(
        self,
        df15: pd.DataFrame,
        task: str,
        *,
        progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        checkpoint_path: Optional[Path] = None,
        resume_path: Optional[Path] = None,
        resume: bool = True,
        dataset_signature: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        base_cfg = deepcopy(self.labels_cfg[task])
        candidates = self._build_candidates(task, base_cfg)
        candidate_map = {
            self._candidate_name(task, patch): patch
            for patch in candidates
        }
        rows_by_name: Dict[str, Dict[str, Any]] = {}

        if resume:
            existing_df = self._load_candidate_frame(resume_path or checkpoint_path)
            existing_df = self._filter_candidate_frame_by_dataset_signature(
                existing_df,
                dataset_signature=dataset_signature,
            )
            if not existing_df.empty and "candidate_name" in existing_df.columns:
                for rec in existing_df.to_dict(orient="records"):
                    cname = str(rec.get("candidate_name") or "")
                    if cname in candidate_map:
                        rows_by_name[cname] = rec

        console_stage(
            f"Tuning {task}",
            f"candidates={len(candidates)} cached={len(rows_by_name)}",
            status="info",
        )

        for idx, patch in enumerate(candidates, start=1):
            candidate_name = self._candidate_name(task, patch)
            if resume and candidate_name in rows_by_name:
                continue
            cfg = deepcopy(base_cfg)
            cfg.update(patch)
            label_series, hazard_time = self._compute_task_labels(df15, task, cfg)
            positive_rate = float(pd.Series(label_series).mean()) if len(label_series) else 0.0
            cv_ap, cv_auc = self._baseline_predictability(df15, label_series)
            objective = self._objective_score(positive_rate, cv_ap, cv_auc)
            row = {
                "task": task,
                "candidate_name": candidate_name,
                "is_baseline": int(self._is_same_candidate(base_cfg, patch)),
                **patch,
                "positive_rate": positive_rate,
                "positives": int(np.sum(label_series)),
                "negatives": int(len(label_series) - np.sum(label_series)),
                "cv_ap": cv_ap,
                "cv_auc": cv_auc,
                "objective_score": objective,
                "dataset_signature": dataset_signature,
            }
            if hazard_time is not None and len(hazard_time):
                row["median_hazard_time"] = float(np.nanmedian(hazard_time))
            rows_by_name[candidate_name] = row
            checkpoint_target = checkpoint_path or resume_path
            if checkpoint_target is not None:
                checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(list(rows_by_name.values())).to_csv(checkpoint_target, index=False)
            if progress_hook is not None:
                progress_hook(
                    {
                        "task": task,
                        "candidate_index": idx,
                        "candidate_total": len(candidates),
                        "candidate_name": row["candidate_name"],
                        "objective_score": row["objective_score"],
                        "cv_ap": row["cv_ap"],
                        "cv_auc": row["cv_auc"],
                        "positive_rate": row["positive_rate"],
                        "is_baseline": bool(row["is_baseline"]),
                    }
                )

        result_df = pd.DataFrame(list(rows_by_name.values()))
        if result_df.empty:
            return pd.DataFrame(), {}
        result_df = result_df.sort_values(
            ["objective_score", "cv_ap", "positive_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        recommended = result_df.iloc[0].to_dict()
        return result_df, recommended

    def _build_candidates(self, task: str, base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = DEFAULT_SWEEPS.get(task, [])
        baseline = {
            key: base_cfg[key]
            for key in ("horizon_bars", "min_horizon", "max_horizon")
            if key in base_cfg
        }
        merged = [baseline, *candidates]
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in merged:
            key = tuple(sorted(item.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _task_partial_path(out_root: Optional[Path], task: str) -> Optional[Path]:
        if out_root is None:
            return None
        return out_root / f"{task}_tuning.partial.csv"

    @staticmethod
    def _task_final_path(out_root: Optional[Path], task: str) -> Optional[Path]:
        if out_root is None:
            return None
        return out_root / f"{task}_tuning.csv"

    def _load_task_checkpoint_frame(
        self,
        out_root: Path,
        task: str,
        *,
        dataset_signature: Optional[str] = None,
    ) -> pd.DataFrame:
        partial = self._task_partial_path(out_root, task)
        final = self._task_final_path(out_root, task)
        for path in (partial, final):
            if path is None or not path.exists():
                continue
            df = self._load_candidate_frame(path)
            df = self._filter_candidate_frame_by_dataset_signature(
                df,
                dataset_signature=dataset_signature,
            )
            if not df.empty:
                return df
        return pd.DataFrame()

    @staticmethod
    def _load_candidate_frame(path: Optional[Path]) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            LOG.warning("[LabelEmpiricalTuner] Failed loading checkpoint %s: %s", path, exc)
            return pd.DataFrame()

    @staticmethod
    def _filter_candidate_frame_by_dataset_signature(
        df: pd.DataFrame,
        *,
        dataset_signature: Optional[str],
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if not dataset_signature:
            return df
        if "dataset_signature" not in df.columns:
            return pd.DataFrame()
        out = df[df["dataset_signature"].astype(str) == str(dataset_signature)].copy()
        return out.reset_index(drop=True)

    def _seed_task_checkpoints_from_progress_events(
        self,
        out_root: Path,
        tasks: Iterable[str],
        *,
        dataset_signature: Optional[str] = None,
    ) -> None:
        events_path = out_root / "progress.ndjson"
        if not events_path.exists():
            return
        task_set = {str(t) for t in tasks}
        rows_by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
        try:
            with events_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    if str(evt.get("event")) != "candidate_scored":
                        continue
                    task = str(evt.get("task") or "")
                    if task not in task_set:
                        continue
                    if dataset_signature and str(evt.get("dataset_signature") or "") != str(dataset_signature):
                        continue
                    candidate_name = str(evt.get("candidate_name") or "")
                    if not candidate_name:
                        continue
                    row = {
                        "task": task,
                        "candidate_name": candidate_name,
                        "is_baseline": int(bool(evt.get("is_baseline", False))),
                        "positive_rate": evt.get("positive_rate"),
                        "cv_ap": evt.get("cv_ap"),
                        "cv_auc": evt.get("cv_auc"),
                        "objective_score": evt.get("objective_score"),
                        "dataset_signature": dataset_signature,
                    }
                    row.update(self._patch_from_candidate_name(candidate_name))
                    rows_by_task.setdefault(task, {})[candidate_name] = row
        except Exception as exc:
            LOG.warning("[LabelEmpiricalTuner] Failed seeding from progress events %s: %s", events_path, exc)
            return

        seeded_tasks = 0
        seeded_rows = 0
        for task in sorted(rows_by_task.keys()):
            partial = self._task_partial_path(out_root, task)
            final = self._task_final_path(out_root, task)
            if (partial is not None and partial.exists()) or (final is not None and final.exists()):
                continue
            task_rows = list(rows_by_task.get(task, {}).values())
            if not task_rows or partial is None:
                continue
            df = pd.DataFrame(task_rows)
            if df.empty:
                continue
            partial.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(partial, index=False)
            seeded_tasks += 1
            seeded_rows += len(df)
        if seeded_tasks > 0:
            console_stage(
                "Resume seed",
                f"seeded {seeded_rows} cached candidates across {seeded_tasks} tasks from progress.ndjson",
                status="info",
            )

    @staticmethod
    def _patch_from_candidate_name(candidate_name: str) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}
        parts = [p.strip() for p in str(candidate_name).split("|")]
        for chunk in parts[1:]:
            if "=" not in chunk:
                continue
            key, raw = chunk.split("=", 1)
            key = key.strip()
            raw = raw.strip()
            patch[key] = LabelEmpiricalTuner._coerce_scalar(raw)
        return patch

    @staticmethod
    def _coerce_scalar(raw: str) -> Any:
        lowered = str(raw).strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if lowered and all(ch not in lowered for ch in (".", "e")):
                return int(lowered)
        except Exception:
            pass
        try:
            return float(lowered)
        except Exception:
            return raw

    def _compute_task_labels(
        self,
        df15: pd.DataFrame,
        task: str,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        fn = LABEL_FUNCTIONS[task]
        if task == "hazard":
            event, tte = fn(df15, cfg)
            return event.astype(int).values, tte.astype(float).values
        labels = fn(df15, cfg)
        return labels.astype(int).values, None

    def _candidate_name(self, task: str, patch: Dict[str, Any]) -> str:
        bits = [task]
        for key, value in sorted(patch.items()):
            bits.append(f"{key}={value}")
        return " | ".join(bits)

    @staticmethod
    def _is_same_candidate(base_cfg: Dict[str, Any], patch: Dict[str, Any]) -> bool:
        for key in ("horizon_bars", "min_horizon", "max_horizon"):
            if key in patch and base_cfg.get(key) != patch.get(key):
                return False
        return True

    def _baseline_predictability(self, df15: pd.DataFrame, y: np.ndarray) -> Tuple[float, float]:
        if len(y) < 50 or np.unique(y).size < 2:
            return 0.0, 0.5

        feature_cols = []
        for col in df15.columns:
            if col in self.EXCLUDE_COLS:
                continue
            if col.startswith("label_") or col.startswith("p_") or col.startswith("prob_") or col.startswith("q_"):
                continue
            if pd.api.types.is_numeric_dtype(df15[col]):
                feature_cols.append(col)

        if not feature_cols:
            return 0.0, 0.5

        X = df15[feature_cols].copy()
        valid_cols = []
        for col in X.columns:
            series = X[col]
            if not series.notna().any():
                continue
            if series.dropna().nunique() <= 1:
                continue
            valid_cols.append(col)
        if not valid_cols:
            return 0.0, 0.5
        X = X[valid_cols]

        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(X.columns),
                )
            ],
            remainder="drop",
        )
        model = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        )

        tscv = TimeSeriesSplit(n_splits=max(2, min(4, len(X) - 1)))
        aps: List[float] = []
        aucs: List[float] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.impute\._base")
            for tr_idx, va_idx in tscv.split(X):
                y_train = y[tr_idx]
                y_valid = y[va_idx]
                if np.unique(y_train).size < 2 or np.unique(y_valid).size < 2:
                    continue
                model.fit(X.iloc[tr_idx], y_train)
                prob = model.predict_proba(X.iloc[va_idx])[:, 1]
                aps.append(float(average_precision_score(y_valid, prob)))
                aucs.append(float(roc_auc_score(y_valid, prob)))

        if not aps:
            return 0.0, 0.5
        return float(np.mean(aps)), float(np.mean(aucs))

    @staticmethod
    def _objective_score(positive_rate: float, cv_ap: float, cv_auc: float) -> float:
        # Prefer labels that are learnable but not pathologically sparse/dense.
        balance_floor = min(positive_rate / 0.05, 1.0) if positive_rate > 0 else 0.0
        balance_cap = min((1.0 - positive_rate) / 0.50, 1.0) if positive_rate < 1 else 0.0
        balance = max(0.0, min(balance_floor, balance_cap))
        return (0.65 * cv_ap) + (0.20 * max(cv_auc - 0.5, 0.0)) + (0.15 * balance)

    @staticmethod
    def _json_default(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _human_duration(seconds: Optional[float]) -> str:
        if seconds is None:
            return "-"
        total = max(int(seconds), 0)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _dataset_signature(df15: pd.DataFrame) -> str:
        if df15 is None or df15.empty:
            return "rows=0"
        first_dt = ""
        last_dt = ""
        if "dt" in df15.columns:
            dt_series = pd.to_datetime(df15["dt"], errors="coerce").dropna()
            if not dt_series.empty:
                first_dt = dt_series.iloc[0].isoformat()
                last_dt = dt_series.iloc[-1].isoformat()
        return f"rows={len(df15)}|first_dt={first_dt}|last_dt={last_dt}"

    def _write_progress_snapshot(self, path: Optional[Path], payload: Dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, default=self._json_default),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _append_progress_event(self, path: Optional[Path], payload: Dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, default=self._json_default)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
