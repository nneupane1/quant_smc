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
    ) -> Dict[str, Any]:
        selected_tasks = list(tasks or DEFAULT_SWEEPS.keys())
        console_rule("Label Horizon Tuning", style="magenta")
        console_kv(
            "Tuning Plan",
            {
                "rows": fmt_num(len(df15)),
                "tasks": ", ".join(selected_tasks),
                "output_dir": output_dir or "-",
                "auto_promote": auto_promote,
                "min_improvement": min_improvement,
            },
            style="magenta",
        )

        progress_path: Optional[Path] = None
        progress_events_path: Optional[Path] = None
        progress_state: Dict[str, Any] = {}
        started_ts = self._utc_now_iso()
        started_wall = datetime.now(timezone.utc)
        candidate_totals: Dict[str, int] = {}
        for task in selected_tasks:
            if task not in LABEL_FUNCTIONS:
                raise ValueError(f"Unknown label tuning task: {task}")
            base_cfg = deepcopy(self.labels_cfg[task])
            candidate_totals[task] = len(self._build_candidates(task, base_cfg))
        total_candidates = int(sum(candidate_totals.values()))
        console_kv(
            "Task Candidate Counts",
            {**{task: candidate_totals[task] for task in selected_tasks}, "total_candidates": total_candidates},
            style="cyan",
        )
        if output_dir:
            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            progress_path = out_root / "progress.json"
            progress_events_path = out_root / "progress.ndjson"

            progress_state = {
                "status": "running",
                "started_at": started_ts,
                "updated_at": started_ts,
                "rows": int(len(df15)),
                "tasks_total": int(len(selected_tasks)),
                "tasks_completed": 0,
                "task_order": selected_tasks,
                "current_task": None,
                "candidates_total": total_candidates,
                "candidates_completed": 0,
                "candidates_remaining": total_candidates,
                "eta_seconds": None,
                "eta_hms": None,
                "elapsed_seconds": 0,
                "elapsed_hms": "00:00",
                "percent_complete": 0.0,
                "percent_remaining": 100.0,
                "tasks": {
                    task: {
                        "status": "pending",
                        "candidates_total": int(candidate_totals[task]),
                        "candidates_completed": 0,
                    }
                    for task in selected_tasks
                },
            }
            self._write_progress_snapshot(progress_path, progress_state)
            self._append_progress_event(
                progress_events_path,
                {"event": "tuning_started", "timestamp": started_ts, "task_order": selected_tasks},
            )

        results: Dict[str, Any] = {}
        promotions: Dict[str, Dict[str, Any]] = {}
        try:
            for task_idx, task in enumerate(selected_tasks, start=1):
                if task not in LABEL_FUNCTIONS:
                    raise ValueError(f"Unknown label tuning task: {task}")
                console_rule(f"Task {task_idx}/{len(selected_tasks)} | {task}", style="cyan")
                console_stage(
                    "Task started",
                    f"candidates={candidate_totals[task]}",
                    status="info",
                )

                if progress_path is not None:
                    progress_state["current_task"] = task
                    progress_state["updated_at"] = self._utc_now_iso()
                    progress_state["tasks"][task]["status"] = "running"
                    self._write_progress_snapshot(progress_path, progress_state)

                def _progress_hook(evt: Dict[str, Any]) -> None:
                    if progress_path is None:
                        return
                    progress_state["updated_at"] = self._utc_now_iso()
                    progress_state["candidates_completed"] = int(progress_state["candidates_completed"]) + 1
                    tstate = progress_state["tasks"][task]
                    tstate["candidates_completed"] = int(evt.get("candidate_index", tstate["candidates_completed"]))
                    tstate["candidate_name"] = evt.get("candidate_name")
                    tstate["last_objective_score"] = evt.get("objective_score")
                    tstate["last_cv_ap"] = evt.get("cv_ap")
                    tstate["last_cv_auc"] = evt.get("cv_auc")
                    tstate["last_positive_rate"] = evt.get("positive_rate")

                    done = float(progress_state["candidates_completed"])
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
                            "global_candidates_completed": progress_state["candidates_completed"],
                            "global_candidates_total": progress_state["candidates_total"],
                            "eta_seconds": eta,
                        },
                    )

                result_df, recommended = self._tune_task(df15, task, progress_hook=_progress_hook)
                baseline = result_df[result_df["is_baseline"] == 1].iloc[0].to_dict()
                improvement = float(recommended["objective_score"]) - float(baseline["objective_score"])
                should_promote = (
                    bool(auto_promote)
                    and bool(recommended["is_baseline"] == 0)
                    and improvement >= float(min_improvement)
                )
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
                    progress_state["tasks_completed"] = int(progress_state["tasks_completed"]) + 1
                    progress_state["current_task"] = None
                    progress_state["updated_at"] = self._utc_now_iso()
                    progress_state["tasks"][task]["status"] = "done"
                    progress_state["tasks"][task]["recommended"] = recommended.get("candidate_name")
                    progress_state["tasks"][task]["improvement_vs_baseline"] = improvement
                    progress_state["tasks"][task]["promoted"] = bool(should_promote)
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
                        },
                    )
                tasks_left = len(selected_tasks) - task_idx
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
                },
            )

        return results

    def _tune_task(
        self,
        df15: pd.DataFrame,
        task: str,
        *,
        progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        base_cfg = deepcopy(self.labels_cfg[task])
        candidates = self._build_candidates(task, base_cfg)
        rows: List[Dict[str, Any]] = []
        console_stage(f"Tuning {task}", f"candidates={len(candidates)}", status="info")

        for idx, patch in enumerate(candidates, start=1):
            cfg = deepcopy(base_cfg)
            cfg.update(patch)
            label_series, hazard_time = self._compute_task_labels(df15, task, cfg)
            positive_rate = float(pd.Series(label_series).mean()) if len(label_series) else 0.0
            cv_ap, cv_auc = self._baseline_predictability(df15, label_series)
            objective = self._objective_score(positive_rate, cv_ap, cv_auc)
            row = {
                "task": task,
                "candidate_name": self._candidate_name(task, patch),
                "is_baseline": int(self._is_same_candidate(base_cfg, patch)),
                **patch,
                "positive_rate": positive_rate,
                "positives": int(np.sum(label_series)),
                "negatives": int(len(label_series) - np.sum(label_series)),
                "cv_ap": cv_ap,
                "cv_auc": cv_auc,
                "objective_score": objective,
            }
            if hazard_time is not None and len(hazard_time):
                row["median_hazard_time"] = float(np.nanmedian(hazard_time))
            rows.append(row)
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

        result_df = pd.DataFrame(rows).sort_values(
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
