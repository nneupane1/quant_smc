from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.training.model_trainer import ModelTrainer
from quant_system.utils.logger import console_kv, console_stage, fmt_num, get_logger

LOG = get_logger("tcn_stack_trainer")


TCN_STACK_TARGETS = {
    "meta_model": "label_liq_flow",
    "confluence_model": "label_liq_flow",
}


class TCNStackTrainer:
    """
    Train stack models over already-trained TCN specialists.

    This intentionally reuses the tree stack challenger logic so routing can compare
    tree-family and TCN-family stacks under the same tabular meta-feature regime.
    """

    def __init__(
        self,
        *,
        asset: str,
        target: str,
        config_loader: ConfigLoader,
        registry: ModelRegistry,
        active_slot: str = "production",
        artifact_root: Optional[Path] = None,
    ):
        if target not in TCN_STACK_TARGETS:
            raise ValueError(
                f"Unsupported TCN stack target '{target}'. "
                f"Choose from: {', '.join(sorted(TCN_STACK_TARGETS))}"
            )
        self.asset = str(asset)
        self.target = str(target)
        self.target_key = TCN_STACK_TARGETS[target]
        self.cfg_loader = config_loader
        self.registry = registry
        self.active_slot = str(active_slot)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.model_trainer = ModelTrainer(config_loader, registry)

        models_yaml = config_loader.load_yaml("models.yaml") or {}
        models_block = (models_yaml.get("models", {}) or {})
        self.stack_cfg = deepcopy(models_block.get(self.target, {}) or {})
        self.inputs = list(self.stack_cfg.get("specialist_inputs", ModelTrainer.SPECIALIST_MODELS))
        if self.artifact_root is not None:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            self.stack_cfg.setdefault(
                "hpo_storage_template",
                f"sqlite:///{(self.artifact_root / 'optuna_{name}.db').resolve()}",
            )
            self.stack_cfg.setdefault("hpo_study_name_template", f"{self.asset}_{{name}}")
            self.stack_cfg.setdefault("hpo_resume", True)

    def required_specialists(self) -> List[str]:
        return list(self.inputs)

    def tree_manifest_path(self) -> Path:
        return Path("artifacts/train") / self.asset / self.target / "train_manifest.json"

    def _tree_baseline_cv_score(self) -> Optional[float]:
        manifest_path = self.tree_manifest_path()
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = ((payload.get("metrics", {}) or {}).get("by_model", {}) or {}).get(self.target, {})
            score = row.get("cv_score")
            if isinstance(score, (int, float)):
                return float(score)
        except Exception:
            return None
        return None

    def _load_specialist_bundle(self, specialist: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        requested_model = f"{specialist}_tcn"
        candidates = [f"{self.asset}_{specialist}_tcn", requested_model]
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                clf, _cal, cfg, meta = self.registry.load_preferred_bundle(
                    candidate,
                    requested_model=requested_model,
                    slot=self.active_slot,
                )
                return clf, cfg, meta
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise FileNotFoundError(f"Missing TCN specialist bundle for '{specialist}'.")

    @staticmethod
    def _batch_positive_proba(model: Any, X_df: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X_df)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        return arr[:, 1].astype(float)

    def build_meta_frame(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        meta_cols: Dict[str, np.ndarray] = {}
        specialist_meta: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        for specialist in self.inputs:
            try:
                clf, cfg, route_meta = self._load_specialist_bundle(specialist)
            except Exception as exc:
                missing.append(f"{specialist}_tcn ({exc})")
                continue

            feature_cols = list(cfg.get("features", []) or cfg.get("feature_cols", []) or [])
            if not feature_cols:
                raise KeyError(f"No feature list found in TCN specialist config for '{specialist}_tcn'.")
            absent = [col for col in feature_cols if col not in df.columns]
            if absent:
                raise KeyError(
                    f"Training frame missing features required by '{specialist}_tcn': {absent[:8]}"
                )

            p = self._batch_positive_proba(clf, df.loc[:, feature_cols].copy())
            meta_cols[f"p_{specialist}"] = p
            specialist_meta[specialist] = {
                "requested_model": f"{specialist}_tcn",
                "resolved_model": route_meta.get("model_id") or f"{specialist}_tcn",
                "version": route_meta.get("version"),
                "selection_source": route_meta.get("selection_source"),
                "selected_metric": route_meta.get("selected_metric"),
                "selected_metric_value": route_meta.get("selected_metric_value"),
                "feature_count": len(feature_cols),
            }

        if missing:
            detail = "; ".join(missing)
            raise FileNotFoundError(
                "Missing required TCN specialists for stack training: "
                f"{detail}"
            )

        X_meta_df = pd.DataFrame(meta_cols, index=df.index)
        return X_meta_df, specialist_meta

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.target_key not in df.columns:
            raise KeyError(f"Training frame missing stack target column '{self.target_key}'.")

        console_kv(
            "TCN Stack Plan",
            {
                "asset": self.asset,
                "target": f"{self.target}_tcn",
                "stack_target_label": self.target_key,
                "specialist_inputs": ", ".join(self.inputs),
                "rows": fmt_num(len(df)),
            },
            style="bright_magenta",
        )

        X_meta_df, specialist_meta = self.build_meta_frame(df)
        y = df[self.target_key].astype(int).values

        console_stage(
            "TCN stack frame ready",
            f"rows={fmt_num(len(X_meta_df))} features={len(X_meta_df.columns)}",
            status="ok",
        )

        started = time.perf_counter()
        stack_name = f"{self.asset}_{self.target}_tcn"
        model, metrics = self.model_trainer._train_with_challengers(
            X_df=X_meta_df,
            y=y,
            cfg=self.stack_cfg,
            name=stack_name,
            num_cols=list(X_meta_df.columns),
            cat_cols=[],
            default_algo="logistic",
        )

        metrics = dict(metrics or {})
        tree_baseline = self._tree_baseline_cv_score()
        metrics["tree_baseline_cv_score"] = tree_baseline
        metrics["delta_vs_tree_cv_score"] = (
            (float(metrics["cv_score"]) - float(tree_baseline))
            if isinstance(metrics.get("cv_score"), (int, float)) and isinstance(tree_baseline, (int, float))
            else None
        )
        metrics["stack_inputs"] = list(self.inputs)
        metrics["meta_feature_cols"] = list(X_meta_df.columns)
        metrics["specialist_routes"] = specialist_meta
        metrics["rows"] = int(len(X_meta_df))
        metrics["tcn_stack_fit_runtime_sec"] = float(time.perf_counter() - started)

        config = {
            "stack_inputs": list(self.inputs),
            "meta_feature_cols": list(X_meta_df.columns),
            "selected_algorithm": metrics.get("selected_algorithm"),
            "decision_threshold": metrics.get("decision_threshold"),
            "calibration": metrics.get("calibration"),
            "threshold_tuning": metrics.get("threshold_tuning"),
            "challenger_scores": metrics.get("challenger_scores", {}),
            "specialist_routes": specialist_meta,
            "tree_baseline_cv_score": tree_baseline,
        }

        LOG.info(
            "[TCNStackTrainer] target=%s selected_algorithm=%s cv_score=%s delta_vs_tree=%s",
            self.target,
            metrics.get("selected_algorithm"),
            metrics.get("cv_score"),
            metrics.get("delta_vs_tree_cv_score"),
        )

        return {
            "model": model,
            "config": config,
            "metrics": metrics,
            "specialist_routes": specialist_meta,
            "outcome": "checkpoint_saved" if bool(metrics.get("checkpoint_interrupted")) else "trained",
        }
