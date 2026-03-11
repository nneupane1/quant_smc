"""Empirical label-horizon tuning on the engineered 15m feature spine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from quant_system.utils.pandas_compat import ensure_stringmethods_alias

try:
    import optuna
except ImportError:  # pragma: no cover - runtime fallback
    optuna = None

try:
    ensure_stringmethods_alias()
    import lightgbm as lgb
except Exception:  # pragma: no cover - runtime fallback
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - runtime fallback
    xgb = None

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


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Leak-safe numeric clipping transformer used inside CV folds."""

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.lower_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.lower_ is None or self.upper_ is None:
            return arr
        return np.clip(arr, self.lower_, self.upper_)


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
        labels_blob = deepcopy(config_loader.load_yaml("labels.yaml"))
        models_blob = deepcopy(config_loader.load_yaml("models.yaml"))
        self.labels_cfg = deepcopy(labels_blob["labels"])
        self.model_cfg = deepcopy(models_blob.get("models", {}))
        self.preproc_cfg = deepcopy(models_blob.get("training_preprocessing", {}))
        self.tuning_cfg = deepcopy(labels_blob.get("tuning", {}))
        self.proxy_models = self._resolve_proxy_models(self.tuning_cfg.get("proxy_models"))
        self.cv_splits = int(self.tuning_cfg.get("cv_splits", 4))
        self.embargo_bars = int(self.tuning_cfg.get("embargo_bars", 2))
        self.hpo_trials_cfg = deepcopy(self.tuning_cfg.get("model_hpo_trials", {}))
        self.use_hpo = bool(self.tuning_cfg.get("use_hpo", True))
        self.default_hpo_trials = int(self.tuning_cfg.get("default_hpo_trials", 6))
        self.heartbeat_seconds = max(int(self.tuning_cfg.get("heartbeat_seconds", 60)), 5)
        score_cfg = self.tuning_cfg.get("score", {})
        weights = score_cfg.get("weights", {}) if isinstance(score_cfg, dict) else {}
        self.score_weight_ap = float(weights.get("ap", 0.55))
        self.score_weight_auc = float(weights.get("auc", 0.15))
        self.score_weight_balance = float(weights.get("balance", 0.10))
        self.score_weight_topk = float(weights.get("topk_precision", 0.20))
        self.score_use_topk = bool(score_cfg.get("include_precision_at_top_k", True)) if isinstance(score_cfg, dict) else True
        self.score_topk_pct = float(score_cfg.get("top_k_pct", 0.10)) if isinstance(score_cfg, dict) else 0.10
        self.model_params_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        promo_cfg = self.tuning_cfg.get("promotion", {})
        self.require_consensus = bool(promo_cfg.get("require_consensus", True))
        self.min_models_for_consensus = int(
            promo_cfg.get("min_models_for_consensus", max(len(self.proxy_models), 1))
        )
        self.require_positive_ci_delta = bool(promo_cfg.get("require_positive_ci_delta", True))
        self.consensus_eps = float(promo_cfg.get("consensus_eps", 1e-9))
        adaptive_cfg = self.tuning_cfg.get("adaptive_refine", {})
        self.adaptive_refine_enabled = bool(adaptive_cfg.get("enabled", True))
        self.adaptive_refine_tasks = {
            str(task).strip()
            for task in adaptive_cfg.get("tasks", ["liq_flow", "bos_cont", "eop", "edp", "hazard"])
            if str(task).strip()
        }
        self.adaptive_refine_top_k = max(int(adaptive_cfg.get("top_k", 2)), 1)
        self.adaptive_refine_max_extra = max(int(adaptive_cfg.get("max_extra_candidates_per_task", 4)), 0)
        self.adaptive_refine_include_midpoint = bool(adaptive_cfg.get("include_midpoint", True))
        self.adaptive_refine_neighbor_multipliers = self._coerce_int_list(
            adaptive_cfg.get("neighbor_multipliers", [-2, -1, 1, 2]),
            fallback=[-2, -1, 1, 2],
            drop_zero=True,
        )
        self.adaptive_refine_task_steps = self._coerce_task_int_map(adaptive_cfg.get("step_by_task", {}))
        self.adaptive_refine_task_min = self._coerce_task_int_map(adaptive_cfg.get("min_horizon_by_task", {}))
        self.adaptive_refine_task_max = self._coerce_task_int_map(adaptive_cfg.get("max_horizon_by_task", {}))
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
                "proxy_models": ", ".join(self.proxy_models),
                "use_hpo": self.use_hpo,
                "default_hpo_trials": self.default_hpo_trials,
                "cv_splits": self.cv_splits,
                "embargo_bars": self.embargo_bars,
                "consensus_gate": self.require_consensus,
                "consensus_min_models": self.min_models_for_consensus,
                "ci_gate": self.require_positive_ci_delta,
                "adaptive_refine": self.adaptive_refine_enabled,
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
                    if not existing_df.empty and "tuning_schema" not in existing_df.columns:
                        existing_df = pd.DataFrame()
                    if not existing_df.empty:
                        existing_df = existing_df[
                            existing_df["tuning_schema"].astype(str) == self._tuning_schema()
                        ]
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
                    tstate = progress_state["tasks"][task]
                    reported_task_total = int(evt.get("candidate_total", tstate.get("candidates_total", 0)) or 0)
                    if reported_task_total > int(tstate.get("candidates_total", 0)):
                        tstate["candidates_total"] = reported_task_total
                    progress_state["candidates_total"] = int(
                        sum(int(v.get("candidates_total", 0)) for v in progress_state.get("tasks", {}).values())
                    )
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
                    total = max(float(progress_state.get("candidates_total", 0)), 1.0)
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
                    progress_state["candidates_remaining"] = int(
                        max(int(progress_state.get("candidates_total", 0)) - progress_state["candidates_completed"], 0)
                    )

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
                should_promote, gate_info = self._promotion_gate(
                    recommended=recommended,
                    baseline=baseline,
                    auto_promote=auto_promote,
                    min_improvement=min_improvement,
                    improvement=improvement,
                )
                if task_final_path is not None:
                    task_final_path.parent.mkdir(parents=True, exist_ok=True)
                    result_df.to_csv(task_final_path, index=False)
                if task_checkpoint_path is not None and task_checkpoint_path.exists():
                    task_checkpoint_path.unlink()
                actual_task_total = int(len(result_df))
                candidate_totals[task] = actual_task_total
                existing_completed_by_task[task] = actual_task_total
                results[task] = {
                    "results": result_df.to_dict(orient="records"),
                    "recommended": recommended,
                    "baseline": baseline,
                    "improvement_vs_baseline": improvement,
                    "promoted": should_promote,
                    "promotion_gate": gate_info,
                }
                console_stage(
                    f"{task} tuned",
                    (
                        f"recommended={recommended['candidate_name']} "
                        f"objective={recommended['objective_score']:.4f} "
                        f"baseline={baseline['objective_score']:.4f} "
                        f"delta={improvement:.4f} "
                        f"promote={'yes' if should_promote else 'no'} "
                        f"gate={gate_info.get('reason', '-')}"
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
                    progress_state["tasks"][task]["candidates_total"] = int(actual_task_total)
                    progress_state["tasks"][task]["candidates_completed"] = int(actual_task_total)
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
                    progress_state["candidates_total"] = int(
                        sum(int(v.get("candidates_total", 0)) for v in progress_state.get("tasks", {}).values())
                    )
                    total = max(float(progress_state.get("candidates_total", 0)), 1.0)
                    progress_state["percent_complete"] = float(global_done / total)
                    progress_state["percent_remaining"] = float(1.0 - progress_state["percent_complete"])
                    progress_state["candidates_remaining"] = int(
                        max(int(progress_state.get("candidates_total", 0)) - global_done, 0)
                    )
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
                candidates_left = (
                    max(int(progress_state.get("candidates_total", 0)) - int(progress_state.get("candidates_completed", 0)), 0)
                    if progress_state
                    else 0
                )
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
                    "promotion_gate": payload.get("promotion_gate", {}),
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
                            "promotion_gate": results[task].get("promotion_gate", {}),
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
        coarse_candidates = self._build_candidates(task, base_cfg)
        candidates = list(coarse_candidates)
        candidate_names = {self._candidate_name(task, patch) for patch in candidates}
        rows_by_name: Dict[str, Dict[str, Any]] = {}

        if resume:
            existing_df = self._load_candidate_frame(resume_path or checkpoint_path)
            existing_df = self._filter_candidate_frame_by_dataset_signature(
                existing_df,
                dataset_signature=dataset_signature,
            )
            if not existing_df.empty and "tuning_schema" not in existing_df.columns:
                console_stage(
                    "Checkpoint schema mismatch",
                    f"{task}: legacy cached rows ignored (missing tuning_schema)",
                    status="warn",
                )
                existing_df = pd.DataFrame()
            if not existing_df.empty:
                existing_df = existing_df[existing_df["tuning_schema"].astype(str) == self._tuning_schema()]
            if not existing_df.empty and "candidate_name" in existing_df.columns:
                for rec in existing_df.to_dict(orient="records"):
                    cname = str(rec.get("candidate_name") or "")
                    if not cname:
                        continue
                    rec_task = str(rec.get("task") or "")
                    if rec_task and rec_task != task:
                        continue
                    rows_by_name[cname] = rec

        console_stage(
            f"Tuning {task}",
            f"candidates={len(candidates)} cached={len(rows_by_name)}",
            status="info",
        )

        X_df, valid_cols = self._prepare_feature_matrix(df15)
        base_labels, _ = self._compute_task_labels(df15, task, base_cfg)
        base_labels = base_labels.astype(int)
        base_positive_rate = float(pd.Series(base_labels).mean()) if len(base_labels) else 0.0
        task_model_params = self._resolve_task_model_params(
            task=task,
            X_df=X_df,
            y=base_labels,
            purge_bars=self._label_span_from_cfg(base_cfg),
            positive_rate=base_positive_rate,
            dataset_signature=dataset_signature,
        )

        def _scored_in_scope() -> int:
            return int(sum(1 for name in candidate_names if name in rows_by_name))

        def _score_candidates(candidate_list: List[Dict[str, Any]], phase: str) -> None:
            for patch in candidate_list:
                candidate_name = self._candidate_name(task, patch)
                if candidate_name not in candidate_names:
                    candidates.append(patch)
                    candidate_names.add(candidate_name)
                if resume and candidate_name in rows_by_name:
                    continue
                done_before = _scored_in_scope()
                candidate_started = time.perf_counter()
                console_stage(
                    "Candidate",
                    f"{task} {done_before + 1}/{len(candidates)} start | {candidate_name} [{phase}]",
                    status="info",
                )
                cfg = deepcopy(base_cfg)
                cfg.update(patch)
                label_series, hazard_time = self._compute_task_labels(df15, task, cfg)
                positive_rate = float(pd.Series(label_series).mean()) if len(label_series) else 0.0
                console_stage(
                    "Candidate labels",
                    (
                        f"{task} {done_before + 1}/{len(candidates)} "
                        f"positives={fmt_num(int(np.sum(label_series)))} "
                        f"rate={positive_rate:.4f}"
                    ),
                    status="info",
                )
                eval_pack = self._baseline_predictability(
                    X_df,
                    label_series,
                    task=task,
                    purge_bars=self._label_span_from_cfg(cfg),
                    positive_rate=positive_rate,
                    model_param_overrides=task_model_params,
                )
                primary = eval_pack["primary"]
                consensus = eval_pack["consensus"]
                row = {
                    "task": task,
                    "candidate_name": candidate_name,
                    "is_baseline": int(self._is_same_candidate(base_cfg, patch)),
                    **patch,
                    "search_phase": phase,
                    "positive_rate": positive_rate,
                    "positives": int(np.sum(label_series)),
                    "negatives": int(len(label_series) - np.sum(label_series)),
                    "cv_ap": float(primary.get("cv_ap", 0.0)),
                    "cv_auc": float(primary.get("cv_auc", 0.5)),
                    "objective_score": float(consensus.get("objective_score", 0.0)),
                    "objective_ci95_low": float(consensus.get("objective_ci95_low", 0.0)),
                    "objective_ci95_high": float(consensus.get("objective_ci95_high", 0.0)),
                    "consensus_wins": int(consensus.get("consensus_wins", 0)),
                    "model_count": int(consensus.get("model_count", 0)),
                    "primary_model": str(primary.get("model_name", "")),
                    "label_span_bars": int(self._label_span_from_cfg(cfg)),
                    "dataset_signature": dataset_signature,
                    "tuning_schema": self._tuning_schema(),
                }
                for model_name, metrics in eval_pack.get("by_model", {}).items():
                    row[f"{model_name}_cv_ap"] = float(metrics.get("cv_ap", 0.0))
                    row[f"{model_name}_cv_auc"] = float(metrics.get("cv_auc", 0.5))
                    row[f"{model_name}_objective_score"] = float(metrics.get("objective_score", 0.0))
                    row[f"{model_name}_objective_ci95_low"] = float(metrics.get("objective_ci95_low", 0.0))
                    row[f"{model_name}_objective_ci95_high"] = float(metrics.get("objective_ci95_high", 0.0))
                    row[f"{model_name}_fold_count"] = int(metrics.get("fold_count", 0))
                if hazard_time is not None and len(hazard_time):
                    row["median_hazard_time"] = float(np.nanmedian(hazard_time))
                rows_by_name[candidate_name] = row
                done_after = _scored_in_scope()
                console_stage(
                    "Candidate done",
                    (
                        f"{task} {done_after}/{len(candidates)} objective={row['objective_score']:.4f} "
                        f"elapsed={self._human_duration(time.perf_counter() - candidate_started)}"
                    ),
                    status="ok",
                )
                checkpoint_target = checkpoint_path or resume_path
                if checkpoint_target is not None:
                    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(list(rows_by_name.values())).to_csv(checkpoint_target, index=False)
                if progress_hook is not None:
                    progress_hook(
                        {
                            "task": task,
                            "candidate_index": done_after,
                            "candidate_total": len(candidates),
                            "candidate_name": row["candidate_name"],
                            "objective_score": row["objective_score"],
                            "cv_ap": row["cv_ap"],
                            "cv_auc": row["cv_auc"],
                            "positive_rate": row["positive_rate"],
                            "is_baseline": bool(row["is_baseline"]),
                            "objective_ci95_low": row["objective_ci95_low"],
                            "objective_ci95_high": row["objective_ci95_high"],
                            "consensus_wins": row["consensus_wins"],
                            "model_count": row["model_count"],
                            "search_phase": phase,
                            "tuning_schema": row["tuning_schema"],
                        }
                    )

        _score_candidates(coarse_candidates, phase="coarse")

        refine_candidates = self._build_refined_horizon_candidates(
            task=task,
            base_cfg=base_cfg,
            coarse_candidates=coarse_candidates,
            rows_by_name=rows_by_name,
        )
        if refine_candidates:
            for patch in refine_candidates:
                cname = self._candidate_name(task, patch)
                if cname in candidate_names:
                    continue
                candidates.append(patch)
                candidate_names.add(cname)
            console_stage(
                "Adaptive refine",
                f"{task}: added {len(refine_candidates)} local horizon candidates",
                status="info",
            )
            _score_candidates(refine_candidates, phase="refine")

        result_df = pd.DataFrame(list(rows_by_name.values()))
        if result_df.empty:
            return pd.DataFrame(), {}
        result_df = self._append_consensus_votes(result_df)
        result_df = result_df.sort_values(
            ["objective_score", "consensus_wins", "objective_ci95_low", "cv_ap", "positive_rate"],
            ascending=[False, False, False, False, False],
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

    def _build_refined_horizon_candidates(
        self,
        *,
        task: str,
        base_cfg: Dict[str, Any],
        coarse_candidates: List[Dict[str, Any]],
        rows_by_name: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.adaptive_refine_enabled or task not in self.adaptive_refine_tasks:
            return []

        coarse_horizons = sorted(
            {
                int(patch.get("horizon_bars"))
                for patch in coarse_candidates
                if "horizon_bars" in patch and pd.notna(patch.get("horizon_bars"))
            }
        )
        if not coarse_horizons:
            return []

        scored_rows: List[Tuple[float, int]] = []
        existing_horizons = set(coarse_horizons)
        if "horizon_bars" in base_cfg and pd.notna(base_cfg.get("horizon_bars")):
            try:
                existing_horizons.add(int(base_cfg.get("horizon_bars")))
            except Exception:
                pass

        for row in rows_by_name.values():
            try:
                row_task = str(row.get("task") or task)
            except Exception:
                row_task = task
            if row_task != task:
                continue
            raw_h = row.get("horizon_bars")
            if raw_h is None or pd.isna(raw_h):
                continue
            try:
                horizon = int(round(float(raw_h)))
            except Exception:
                continue
            existing_horizons.add(horizon)
            try:
                score = float(row.get("objective_score", 0.0))
            except Exception:
                score = 0.0
            scored_rows.append((score, horizon))

        if not scored_rows:
            return []

        scored_rows.sort(key=lambda x: x[0], reverse=True)
        seed_horizons: List[int] = []
        for _, horizon in scored_rows:
            if horizon in seed_horizons:
                continue
            seed_horizons.append(horizon)
            if len(seed_horizons) >= self.adaptive_refine_top_k:
                break
        if not seed_horizons:
            return []

        diffs = [
            int(abs(b - a))
            for a, b in zip(coarse_horizons[:-1], coarse_horizons[1:])
            if int(abs(b - a)) > 0
        ]
        step = int(self.adaptive_refine_task_steps.get(task, min(diffs) if diffs else 1))
        step = max(step, 1)

        min_bound = int(self.adaptive_refine_task_min.get(task, max(1, min(coarse_horizons) - step)))
        max_bound = int(self.adaptive_refine_task_max.get(task, max(coarse_horizons) + step))
        if min_bound > max_bound:
            min_bound, max_bound = max_bound, min_bound

        candidate_horizons: List[int] = []
        for seed in seed_horizons:
            for mult in self.adaptive_refine_neighbor_multipliers:
                cand = int(seed + (mult * step))
                if cand < min_bound or cand > max_bound:
                    continue
                if cand in existing_horizons or cand in candidate_horizons:
                    continue
                candidate_horizons.append(cand)

        if self.adaptive_refine_include_midpoint and len(seed_horizons) >= 2:
            midpoint = int(round((seed_horizons[0] + seed_horizons[1]) / 2.0))
            if (
                midpoint >= min_bound
                and midpoint <= max_bound
                and midpoint not in existing_horizons
                and midpoint not in candidate_horizons
            ):
                candidate_horizons.append(midpoint)

        if not candidate_horizons:
            return []

        best_seed = int(seed_horizons[0])
        candidate_horizons = sorted(candidate_horizons, key=lambda h: (abs(h - best_seed), h))
        if self.adaptive_refine_max_extra > 0:
            candidate_horizons = candidate_horizons[: self.adaptive_refine_max_extra]
        return [{"horizon_bars": int(h)} for h in candidate_horizons]

    @staticmethod
    def _coerce_task_int_map(raw: Any) -> Dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, int] = {}
        for key, value in raw.items():
            try:
                out[str(key).strip()] = int(value)
            except Exception:
                continue
        return out

    @staticmethod
    def _coerce_int_list(
        raw: Any,
        *,
        fallback: List[int],
        drop_zero: bool = False,
    ) -> List[int]:
        source = raw if isinstance(raw, list) else fallback
        out: List[int] = []
        for val in source:
            try:
                iv = int(val)
            except Exception:
                continue
            if drop_zero and iv == 0:
                continue
            if iv not in out:
                out.append(iv)
        return out or list(fallback)

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
                        "objective_ci95_low": evt.get("objective_ci95_low", evt.get("objective_score")),
                        "objective_ci95_high": evt.get("objective_ci95_high", evt.get("objective_score")),
                        "consensus_wins": evt.get("consensus_wins", 0),
                        "model_count": evt.get("model_count", 0),
                        "tuning_schema": evt.get("tuning_schema", "legacy_event"),
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

    def _append_consensus_votes(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        score_cols = [f"{m}_objective_score" for m in self.proxy_models if f"{m}_objective_score" in out.columns]
        if not score_cols:
            out["consensus_wins"] = 0
            out["consensus_share"] = 0.0
            return out
        wins = np.zeros(len(out), dtype=int)
        for col in score_cols:
            max_val = float(pd.to_numeric(out[col], errors="coerce").max())
            cur = pd.to_numeric(out[col], errors="coerce").fillna(-np.inf)
            wins += ((max_val - cur) <= float(self.consensus_eps)).astype(int).values
        out["consensus_wins"] = wins
        out["consensus_share"] = wins / max(len(score_cols), 1)
        return out

    def _promotion_gate(
        self,
        *,
        recommended: Dict[str, Any],
        baseline: Dict[str, Any],
        auto_promote: bool,
        min_improvement: float,
        improvement: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not bool(auto_promote):
            return False, {"reason": "auto_promote_disabled"}
        if bool(recommended.get("is_baseline", 0) == 1):
            return False, {"reason": "recommended_is_baseline"}
        if improvement < float(min_improvement):
            return False, {"reason": "min_improvement_not_met", "improvement": improvement}

        wins = int(recommended.get("consensus_wins", 0) or 0)
        if self.require_consensus and wins < int(self.min_models_for_consensus):
            return False, {
                "reason": "consensus_not_met",
                "consensus_wins": wins,
                "min_required": int(self.min_models_for_consensus),
            }

        if self.require_positive_ci_delta:
            rec_low = float(recommended.get("objective_ci95_low", np.nan))
            base_high = float(baseline.get("objective_ci95_high", np.nan))
            if np.isnan(rec_low) or np.isnan(base_high) or rec_low <= base_high:
                return False, {
                    "reason": "ci_delta_not_positive",
                    "rec_ci95_low": rec_low,
                    "base_ci95_high": base_high,
                }

        return True, {"reason": "passed"}

    def _resolve_proxy_models(self, raw_models: Any) -> List[str]:
        default = ["lightgbm", "xgboost", "logreg"]
        if not isinstance(raw_models, list):
            return default
        allowed = {"logreg", "hgb", "lightgbm", "xgboost"}
        out = [str(m).strip().lower() for m in raw_models if str(m).strip().lower() in allowed]
        return out or default

    @staticmethod
    def _is_model_available(model_name: str) -> bool:
        if model_name == "lightgbm":
            return lgb is not None
        if model_name == "xgboost":
            return xgb is not None
        return True

    def _tuning_schema(self) -> str:
        task_steps = ",".join(
            f"{k}:{v}" for k, v in sorted(self.adaptive_refine_task_steps.items())
        ) or "-"
        task_bounds_min = ",".join(
            f"{k}:{v}" for k, v in sorted(self.adaptive_refine_task_min.items())
        ) or "-"
        task_bounds_max = ",".join(
            f"{k}:{v}" for k, v in sorted(self.adaptive_refine_task_max.items())
        ) or "-"
        return (
            f"v4|models={','.join(self.proxy_models)}|hpo={int(self.use_hpo)}|hpo_default={int(self.default_hpo_trials)}"
            f"|cv={int(self.cv_splits)}|embargo={int(self.embargo_bars)}"
            f"|consensus={int(self.min_models_for_consensus)}|ci={int(self.require_positive_ci_delta)}"
            f"|score={self.score_weight_ap:.3f},{self.score_weight_auc:.3f},{self.score_weight_balance:.3f},{self.score_weight_topk:.3f}"
            f"|refine={int(self.adaptive_refine_enabled)}"
            f"|refine_tasks={','.join(sorted(self.adaptive_refine_tasks))}"
            f"|refine_topk={int(self.adaptive_refine_top_k)}|refine_extra={int(self.adaptive_refine_max_extra)}"
            f"|refine_mid={int(self.adaptive_refine_include_midpoint)}"
            f"|refine_mult={','.join(str(x) for x in self.adaptive_refine_neighbor_multipliers)}"
            f"|refine_step={task_steps}|refine_min={task_bounds_min}|refine_max={task_bounds_max}"
        )

    @staticmethod
    def _label_span_from_cfg(cfg: Dict[str, Any]) -> int:
        if "horizon_bars" in cfg:
            return max(int(cfg.get("horizon_bars", 1)), 1)
        if "max_horizon" in cfg:
            return max(int(cfg.get("max_horizon", 1)), 1)
        if "min_horizon" in cfg:
            return max(int(cfg.get("min_horizon", 1)), 1)
        return 1

    def _prepare_feature_matrix(self, df15: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        feature_cols = []
        for col in df15.columns:
            if col in self.EXCLUDE_COLS:
                continue
            if col.startswith("label_") or col.startswith("p_") or col.startswith("prob_") or col.startswith("q_"):
                continue
            if pd.api.types.is_numeric_dtype(df15[col]):
                feature_cols.append(col)
        if not feature_cols:
            return pd.DataFrame(index=df15.index), []

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
            return pd.DataFrame(index=df15.index), []
        return X[valid_cols], valid_cols

    def _default_empty_eval(self, positive_rate: float) -> Dict[str, Any]:
        base_obj = float(self._objective_score(positive_rate, 0.0, 0.5, 0.0))
        return {
            "cv_ap": 0.0,
            "cv_auc": 0.5,
            "cv_topk_precision": 0.0,
            "objective_score": base_obj,
            "objective_ci95_low": base_obj,
            "objective_ci95_high": base_obj,
            "fold_count": 0,
        }

    def _make_purged_splits(self, n_rows: int, purge_bars: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if n_rows < 50:
            return []
        n_splits = max(2, min(int(self.cv_splits), max(n_rows - 1, 2)))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        embargo = max(int(self.embargo_bars), 0)
        purge = max(int(purge_bars), 0)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_idx, va_idx in tscv.split(np.arange(n_rows)):
            cutoff = int(va_idx[0]) - max(embargo, purge)
            tr = tr_idx[tr_idx < cutoff]
            if len(tr) < 50 or len(va_idx) < 10:
                continue
            splits.append((tr, va_idx))
        return splits

    def _resolve_task_model_params(
        self,
        *,
        task: str,
        X_df: pd.DataFrame,
        y: np.ndarray,
        purge_bars: int,
        positive_rate: float,
        dataset_signature: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        ds_sig = str(dataset_signature or "rows=unknown")
        cache_key = f"{task}|{ds_sig}|{self._tuning_schema()}"
        if cache_key in self.model_params_cache:
            return deepcopy(self.model_params_cache[cache_key])

        task_params: Dict[str, Dict[str, Any]] = {}
        splits = self._make_purged_splits(len(X_df), purge_bars)
        for model_name in self.proxy_models:
            if not self._is_model_available(model_name):
                console_stage("Model unavailable", f"{model_name}: skipped for {task}", status="warn")
                continue
            default_params = self._default_params_for_model(task, model_name)
            trials = self._hpo_trials_for_model(model_name)
            model_started = time.perf_counter()
            console_stage(
                "Model prep",
                (
                    f"{task}/{model_name} | rows={fmt_num(len(X_df))} "
                    f"cv_folds={len(splits)} hpo_trials={trials if self.use_hpo else 0}"
                ),
                status="info",
            )
            if (
                self.use_hpo
                and trials > 0
                and optuna is not None
                and len(splits) > 0
                and np.unique(y).size >= 2
            ):
                best = self._optimize_model_params(
                    task=task,
                    model_name=model_name,
                    X_df=X_df,
                    y=y,
                    splits=splits,
                    positive_rate=positive_rate,
                    default_params=default_params,
                    n_trials=trials,
                )
                task_params[model_name] = best
                source = "hpo"
            else:
                task_params[model_name] = default_params
                source = "default"
            console_stage(
                "Model prep done",
                f"{task}/{model_name} source={source} elapsed={self._human_duration(time.perf_counter() - model_started)}",
                status="ok",
            )
        self.model_params_cache[cache_key] = deepcopy(task_params)
        return task_params

    def _hpo_trials_for_model(self, model_name: str) -> int:
        if isinstance(self.hpo_trials_cfg, dict) and model_name in self.hpo_trials_cfg:
            try:
                return max(int(self.hpo_trials_cfg.get(model_name, 0)), 0)
            except Exception:
                return 0
        return max(int(self.default_hpo_trials), 0)

    def _extract_hpo_space(self, task: str) -> Dict[str, Any]:
        cfg = self.model_cfg.get(task, {})
        return cfg.get("hpo_space", {}) if isinstance(cfg, dict) else {}

    @staticmethod
    def _range_midpoint(v: Any, default: float, as_int: bool = False) -> float | int:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = v
            try:
                val = (float(lo) + float(hi)) / 2.0
            except Exception:
                val = float(default)
        else:
            try:
                val = float(v)
            except Exception:
                val = float(default)
        if as_int:
            return int(round(val))
        return float(val)

    @staticmethod
    def _range_bounds(v: Any, default_lo: float, default_hi: float, as_int: bool = False) -> Tuple[float | int, float | int]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                lo = float(v[0])
                hi = float(v[1])
            except Exception:
                lo = float(default_lo)
                hi = float(default_hi)
        else:
            lo = float(default_lo)
            hi = float(default_hi)
        if lo > hi:
            lo, hi = hi, lo
        if as_int:
            return int(round(lo)), int(round(hi))
        return float(lo), float(hi)

    def _default_params_for_model(self, task: str, model_name: str) -> Dict[str, Any]:
        space = self._extract_hpo_space(task)
        if model_name == "lightgbm":
            return {
                "n_estimators": self._range_midpoint(space.get("n_estimators", [200, 600]), 300, as_int=True),
                "num_leaves": self._range_midpoint(space.get("num_leaves", [31, 255]), 63, as_int=True),
                "max_depth": self._range_midpoint(space.get("max_depth", [3, 12]), 6, as_int=True),
                "learning_rate": self._range_midpoint(space.get("learning_rate", [0.01, 0.15]), 0.05),
                "subsample": self._range_midpoint(space.get("subsample", [0.7, 1.0]), 0.8),
                "colsample_bytree": self._range_midpoint(space.get("colsample_bytree", [0.7, 1.0]), 0.8),
                "reg_alpha": self._range_midpoint(space.get("reg_alpha", [0.0, 5.0]), 0.0),
                "reg_lambda": self._range_midpoint(space.get("reg_lambda", [0.0, 5.0]), 0.0),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": self._range_midpoint(space.get("n_estimators", [200, 600]), 300, as_int=True),
                "max_depth": self._range_midpoint(space.get("max_depth", [3, 12]), 6, as_int=True),
                "learning_rate": self._range_midpoint(space.get("learning_rate", [0.01, 0.15]), 0.05),
                "subsample": self._range_midpoint(space.get("subsample", [0.7, 1.0]), 0.8),
                "colsample_bytree": self._range_midpoint(space.get("colsample_bytree", [0.7, 1.0]), 0.8),
                "reg_alpha": self._range_midpoint(space.get("reg_alpha", [0.0, 5.0]), 0.0),
                "reg_lambda": self._range_midpoint(space.get("reg_lambda", [0.0, 5.0]), 1.0),
                "min_child_weight": self._range_midpoint(space.get("min_child_weight", [1, 8]), 1.0),
            }
        if model_name == "hgb":
            return {
                "max_depth": 6,
                "max_iter": 200,
                "learning_rate": 0.05,
                "min_samples_leaf": 40,
            }
        # logreg
        return {
            "C": self._range_midpoint(space.get("C", [0.01, 10.0]), 1.0),
            "max_iter": 2000,
        }

    def _sample_params_for_model(self, trial, task: str, model_name: str, default_params: Dict[str, Any]) -> Dict[str, Any]:
        space = self._extract_hpo_space(task)
        if model_name == "lightgbm":
            n_estimators_lo, n_estimators_hi = self._range_bounds(space.get("n_estimators"), 120, 600, as_int=True)
            num_leaves_lo, num_leaves_hi = self._range_bounds(space.get("num_leaves"), 15, 255, as_int=True)
            max_depth_lo, max_depth_hi = self._range_bounds(space.get("max_depth"), 3, 12, as_int=True)
            lr_lo, lr_hi = self._range_bounds(space.get("learning_rate"), 0.005, 0.15, as_int=False)
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    int(n_estimators_lo),
                    int(n_estimators_hi),
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    int(num_leaves_lo),
                    int(num_leaves_hi),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    int(max_depth_lo),
                    int(max_depth_hi),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    float(lr_lo),
                    float(lr_hi),
                    log=True,
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 8.0),
            }
        if model_name == "xgboost":
            n_estimators_lo, n_estimators_hi = self._range_bounds(space.get("n_estimators"), 120, 700, as_int=True)
            max_depth_lo, max_depth_hi = self._range_bounds(space.get("max_depth"), 3, 12, as_int=True)
            lr_lo, lr_hi = self._range_bounds(space.get("learning_rate"), 0.005, 0.15, as_int=False)
            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    int(n_estimators_lo),
                    int(n_estimators_hi),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    int(max_depth_lo),
                    int(max_depth_hi),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    float(lr_lo),
                    float(lr_hi),
                    log=True,
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 8.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            }
        if model_name == "hgb":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "max_iter": trial.suggest_int("max_iter", 120, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            }
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "max_iter": int(default_params.get("max_iter", 2000)),
        }

    def _optimize_model_params(
        self,
        *,
        task: str,
        model_name: str,
        X_df: pd.DataFrame,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        positive_rate: float,
        default_params: Dict[str, Any],
        n_trials: int,
    ) -> Dict[str, Any]:
        if optuna is None or n_trials <= 0:
            return default_params

        hpo_started = time.perf_counter()

        def objective(trial):
            params = self._sample_params_for_model(trial, task, model_name, default_params)
            metrics = self._evaluate_proxy_model(
                model_name=model_name,
                params=params,
                X=X_df,
                y=y,
                splits=splits,
                positive_rate=positive_rate,
            )
            return -float(metrics.get("objective_score", 0.0))

        def _on_trial(study, trial):  # pragma: no cover - runtime telemetry
            done = len(study.trials)
            best_obj = -float(study.best_value) if study.best_trial is not None else 0.0
            elapsed = time.perf_counter() - hpo_started
            eta = None
            if done > 0:
                rate = elapsed / done
                eta = max(int((n_trials - done) * rate), 0)
            console_stage(
                "HPO",
                (
                    f"{task}/{model_name} trial {done}/{n_trials} "
                    f"best_objective={best_obj:.4f} "
                    f"elapsed={self._human_duration(elapsed)} "
                    f"eta={self._human_duration(eta)}"
                ),
                status="info",
            )

        try:
            try:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            except Exception:
                pass
            study = optuna.create_study(direction="minimize")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                study.optimize(
                    objective,
                    n_trials=int(n_trials),
                    callbacks=[_on_trial],
                    show_progress_bar=False,
                )
            best = dict(default_params)
            best.update(study.best_params)
            return best
        except Exception as exc:
            LOG.warning("[LabelEmpiricalTuner] HPO failed for task=%s model=%s: %s", task, model_name, exc)
            return default_params

    def _baseline_predictability(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        task: str,
        purge_bars: int,
        positive_rate: float,
        model_param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if len(y) < 50 or np.unique(y).size < 2 or X_df is None or X_df.empty:
            empty = self._default_empty_eval(positive_rate)
            return {
                "primary": {"model_name": self.proxy_models[0], **empty},
                "by_model": {m: dict(empty) for m in self.proxy_models},
                "consensus": {
                    "objective_score": float(empty["objective_score"]),
                    "objective_ci95_low": float(empty["objective_ci95_low"]),
                    "objective_ci95_high": float(empty["objective_ci95_high"]),
                    "consensus_wins": 0,
                    "model_count": len(self.proxy_models),
                },
            }

        splits = self._make_purged_splits(len(X_df), purge_bars)
        if not splits:
            empty = self._default_empty_eval(positive_rate)
            return {
                "primary": {"model_name": self.proxy_models[0], **empty},
                "by_model": {m: dict(empty) for m in self.proxy_models},
                "consensus": {
                    "objective_score": float(empty["objective_score"]),
                    "objective_ci95_low": float(empty["objective_ci95_low"]),
                    "objective_ci95_high": float(empty["objective_ci95_high"]),
                    "consensus_wins": 0,
                    "model_count": len(self.proxy_models),
                },
            }

        by_model: Dict[str, Dict[str, Any]] = {}
        active_models = [m for m in self.proxy_models if self._is_model_available(m)]
        if not active_models:
            active_models = ["logreg"]
        for model_name in active_models:
            params = dict((model_param_overrides or {}).get(model_name, self._default_params_for_model(task, model_name)))
            metrics = self._evaluate_proxy_model(
                model_name=model_name,
                params=params,
                X=X_df,
                y=y,
                splits=splits,
                positive_rate=positive_rate,
            )
            by_model[model_name] = metrics

        primary_name = active_models[0]
        primary = {"model_name": primary_name, **by_model.get(primary_name, self._default_empty_eval(positive_rate))}
        objectives = [float(by_model[m]["objective_score"]) for m in active_models if m in by_model]
        lows = [float(by_model[m]["objective_ci95_low"]) for m in active_models if m in by_model]
        highs = [float(by_model[m]["objective_ci95_high"]) for m in active_models if m in by_model]
        max_obj = max(objectives) if objectives else 0.0
        wins = int(sum(1 for m in active_models if m in by_model and (max_obj - float(by_model[m]["objective_score"])) <= self.consensus_eps))
        consensus = {
            "objective_score": float(np.mean(objectives)) if objectives else 0.0,
            "objective_ci95_low": float(min(lows)) if lows else 0.0,
            "objective_ci95_high": float(max(highs)) if highs else 0.0,
            "consensus_wins": wins,
            "model_count": len(objectives),
        }
        return {"primary": primary, "by_model": by_model, "consensus": consensus}

    def _build_num_preprocessor(self, feature_cols: List[str], model_name: str) -> ColumnTransformer:
        num_imputer = str(self.preproc_cfg.get("num_imputer", "median")).lower()
        scaler_mode = str(self.preproc_cfg.get("scaler", "standard")).lower()
        scale_for_tree = bool(self.preproc_cfg.get("scale_for_tree_models", False))
        clip_enabled = bool(self.preproc_cfg.get("outlier_clip", True))
        clip_q = self.preproc_cfg.get("clip_quantiles", [0.005, 0.995])
        try:
            lower_q = float(clip_q[0])
            upper_q = float(clip_q[1])
        except Exception:
            lower_q, upper_q = 0.005, 0.995
        lower_q = min(max(lower_q, 0.0), 0.49)
        upper_q = max(min(upper_q, 1.0), 0.51)

        num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_imputer))]
        if clip_enabled:
            num_steps.append(("clip", QuantileClipper(lower_q=lower_q, upper_q=upper_q)))
        is_tree = model_name in {"lightgbm", "xgboost", "hgb"}
        if scaler_mode != "none" and (scale_for_tree or not is_tree):
            num_steps.append(("scaler", StandardScaler()))

        return ColumnTransformer(
            transformers=[("num", Pipeline(steps=num_steps), list(feature_cols))],
            remainder="drop",
        )

    def _build_proxy_model(self, model_name: str, feature_cols: List[str], params: Dict[str, Any]) -> Pipeline:
        pre = self._build_num_preprocessor(feature_cols, model_name)

        if model_name == "lightgbm":
            clf = lgb.LGBMClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                num_leaves=int(params.get("num_leaves", 63)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 0.0)),
                objective="binary",
                class_weight="balanced",
                verbosity=-1,
            )
        elif model_name == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                min_child_weight=float(params.get("min_child_weight", 1.0)),
                eval_metric="logloss",
                tree_method="hist",
                objective="binary:logistic",
                scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
                verbosity=0,
                n_jobs=1,
            )
        elif model_name == "hgb":
            clf = HistGradientBoostingClassifier(
                max_depth=int(params.get("max_depth", 6)),
                max_iter=int(params.get("max_iter", 200)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                min_samples_leaf=int(params.get("min_samples_leaf", 40)),
                random_state=42,
            )
        else:
            clf = LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 2000)),
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
            )
        return Pipeline(steps=[("pre", pre), ("clf", clf)])

    def _evaluate_proxy_model(
        self,
        *,
        model_name: str,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        positive_rate: float,
    ) -> Dict[str, Any]:
        aps: List[float] = []
        aucs: List[float] = []
        topks: List[float] = []
        objs: List[float] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.impute\._base")
            for tr_idx, va_idx in splits:
                y_train = y[tr_idx]
                y_valid = y[va_idx]
                if np.unique(y_train).size < 2 or np.unique(y_valid).size < 2:
                    continue
                fold_params = dict(params)
                if model_name == "xgboost" and "scale_pos_weight" not in fold_params:
                    pos = float(np.sum(y_train == 1))
                    neg = float(np.sum(y_train == 0))
                    fold_params["scale_pos_weight"] = (neg / max(pos, 1.0)) if pos > 0 else 1.0
                model = self._build_proxy_model(model_name, list(X.columns), fold_params)
                model.fit(X.iloc[tr_idx], y_train)
                prob = self._positive_class_proba(model, X.iloc[va_idx])
                ap = float(average_precision_score(y_valid, prob))
                try:
                    auc = float(roc_auc_score(y_valid, prob))
                except Exception:
                    auc = 0.5
                topk = self._topk_precision(y_valid, prob, self.score_topk_pct) if self.score_use_topk else 0.0
                aps.append(ap)
                aucs.append(auc)
                topks.append(topk)
                objs.append(float(self._objective_score(positive_rate, ap, auc, topk)))

        if not aps:
            return self._default_empty_eval(positive_rate)

        obj_mean = float(np.mean(objs))
        obj_std = float(np.std(objs, ddof=1)) if len(objs) > 1 else 0.0
        ci_half = 1.96 * obj_std / max(np.sqrt(len(objs)), 1.0) if len(objs) > 1 else 0.0
        return {
            "cv_ap": float(np.mean(aps)),
            "cv_auc": float(np.mean(aucs)),
            "cv_topk_precision": float(np.mean(topks)) if topks else 0.0,
            "objective_score": obj_mean,
            "objective_ci95_low": float(obj_mean - ci_half),
            "objective_ci95_high": float(obj_mean + ci_half),
            "fold_count": int(len(objs)),
        }

    @staticmethod
    def _positive_class_proba(model: Pipeline, X_df: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_df)
            if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
                return np.asarray(p[:, 1], dtype=float)
            return np.asarray(p, dtype=float).reshape(-1)
        if hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(X_df), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-z))
        pred = np.asarray(model.predict(X_df), dtype=float).reshape(-1)
        return np.clip(pred, 0.0, 1.0)

    @staticmethod
    def _topk_precision(y_true: np.ndarray, prob: np.ndarray, top_k_pct: float) -> float:
        if y_true is None or prob is None or len(y_true) == 0:
            return 0.0
        pct = float(min(max(top_k_pct, 0.001), 1.0))
        k = max(int(round(len(y_true) * pct)), 1)
        idx = np.argsort(np.asarray(prob, dtype=float))[-k:]
        return float(np.mean(np.asarray(y_true)[idx]))

    def _objective_score(
        self,
        positive_rate: float,
        cv_ap: float,
        cv_auc: float,
        topk_precision: float = 0.0,
    ) -> float:
        balance_floor = min(positive_rate / 0.05, 1.0) if positive_rate > 0 else 0.0
        balance_cap = min((1.0 - positive_rate) / 0.50, 1.0) if positive_rate < 1 else 0.0
        balance = max(0.0, min(balance_floor, balance_cap))
        auc_edge = max(float(cv_auc) - 0.5, 0.0)
        topk_term = float(topk_precision) if self.score_use_topk else 0.0
        return (
            self.score_weight_ap * float(cv_ap)
            + self.score_weight_auc * auc_edge
            + self.score_weight_balance * balance
            + (self.score_weight_topk * topk_term if self.score_use_topk else 0.0)
        )

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
        # Include schema + sampled content hash so resume caches are invalidated
        # when engineered feature values change while date span stays identical.
        schema_blob = "|".join(f"{c}:{str(df15[c].dtype)}" for c in df15.columns)
        schema_hash = hashlib.sha1(schema_blob.encode("utf-8")).hexdigest()[:12]

        if len(df15) <= 20000:
            sample = df15
        else:
            sample = pd.concat([df15.head(10000), df15.tail(10000)], axis=0, ignore_index=False)
        sample_hash_values = pd.util.hash_pandas_object(sample, index=True).values
        sample_hash = hashlib.sha1(sample_hash_values.tobytes()).hexdigest()[:16]

        return (
            f"rows={len(df15)}|first_dt={first_dt}|last_dt={last_dt}"
            f"|schema={schema_hash}|sample={sample_hash}"
        )

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
