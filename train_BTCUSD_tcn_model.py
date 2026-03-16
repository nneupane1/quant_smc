from __future__ import annotations

import os

# Safer defaults for macOS scientific stack + PyTorch to reduce native-thread crashes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "numpy", "sklearn", "optuna"))

import argparse
from pathlib import Path
import time
from typing import Optional

from quant_system.cli.common import save_json
from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.registry.versioning import ModelVersionManager
from quant_system.train_orchestrator import TrainOrchestrator
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_num, fmt_seconds

ASSET = "BTCUSD"
CONFIG_DIR = "quant_system/config"
TARGET_COLUMN_MAP = {
    "liq_flow": "label_liq_flow",
    "bos_cont": "label_bos_cont",
    "momo": "label_momo",
    "flow_1h": "label_flow_1h",
    "eop": "label_eop",
    "edp": "label_edp",
}


def _default_tf_dir(cfg: dict) -> str:
    return str(Path((cfg.get("paths", {}) or {}).get("tf", "data/tf")))


def _default_features_csv() -> str:
    return f"artifacts/features/{ASSET}/{ASSET}_features.csv"


def _default_labels_csv() -> str:
    return f"artifacts/labels/{ASSET}/{ASSET}_labels.csv"


def _resolve_registry_dir(cfg: dict) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    models_cfg = cfg.get("models", {}) or {}
    registry = (
        paths_cfg.get("model_registry")
        or models_cfg.get("registry_path")
        or "models"
    )
    return Path(str(registry))


def run_target(
    target: str,
    *,
    trials: Optional[int] = None,
    cv_splits: Optional[int] = None,
    adaptive_stop: Optional[bool] = None,
    adaptive_min_completed_trials: Optional[int] = None,
    adaptive_no_improve_trials: Optional[int] = None,
    adaptive_min_delta: Optional[float] = None,
    resume: bool = True,
) -> dict:
    if target not in TARGET_COLUMN_MAP:
        raise SystemExit(f"Unsupported target '{target}'. Choose from: {', '.join(sorted(TARGET_COLUMN_MAP))}")

    started = time.perf_counter()
    persisted = False
    root = Path("artifacts/train") / ASSET / f"{target}_tcn"
    root.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from quant_system.ml.training.tcn_trainer import TCNSpecialistTrainer, merge_tcn_cfg
        except ModuleNotFoundError as exc:
            if str(exc).strip().endswith("No module named 'torch'"):
                raise SystemExit(
                    "PyTorch is required for TCN training.\n"
                    "Install in this environment first:\n"
                    "pip install torch"
                ) from exc
            raise

        loader = ConfigLoader(CONFIG_DIR)
        unified_cfg = loader.load()
        models_yaml = loader.load_yaml("models.yaml") or {}
        models_block = (models_yaml.get("models", {}) or {})
        target_cfg = deepcopy_dict(models_block.get(target, {}))

        tcn_cfg = merge_tcn_cfg(
            models_yaml=models_yaml,
            target=target,
            trials_override=trials,
            cv_override=cv_splits,
        )
        if str(tcn_cfg.get("calibrator", "auto")).lower() == "auto" and target_cfg.get("calibrator"):
            tcn_cfg["calibrator"] = str(target_cfg.get("calibrator"))
        if adaptive_stop is not None:
            tcn_cfg["hpo_adaptive_stop"] = bool(adaptive_stop)
        if adaptive_min_completed_trials is not None:
            tcn_cfg["hpo_adaptive_min_completed_trials"] = int(adaptive_min_completed_trials)
        if adaptive_no_improve_trials is not None:
            tcn_cfg["hpo_adaptive_no_improve_trials"] = int(adaptive_no_improve_trials)
        if adaptive_min_delta is not None:
            tcn_cfg["hpo_adaptive_min_delta"] = float(adaptive_min_delta)

        tf_dir = _default_tf_dir(unified_cfg)
        features_csv = _default_features_csv()
        labels_csv = _default_labels_csv()
        merged_out = root / "training_frame.csv"
        model_state_out = root / "model_state.json"
        manifest_out = root / "train_manifest.json"
        tree_manifest = Path("artifacts/train") / ASSET / target / "train_manifest.json"
        hpo_db_path = root / f"optuna_{target}.db"
        if not tcn_cfg.get("hpo_storage"):
            tcn_cfg["hpo_storage"] = f"sqlite:///{hpo_db_path.resolve()}"
        if not tcn_cfg.get("hpo_study_name"):
            tcn_cfg["hpo_study_name"] = f"{ASSET}_{target}_tcn"
        tcn_cfg["artifact_root"] = str(root)
        progress_json = root / str(tcn_cfg.get("progress_snapshot_filename", "hpo_progress.json"))
        progress_ndjson = root / str(tcn_cfg.get("progress_events_filename", "hpo_progress.ndjson"))

        console_rule(f"Train TCN | {ASSET} | {target}", style="bright_magenta")
        console_kv(
            "TCN Plan",
            {
                "asset": ASSET,
                "target": target,
                "label_col": TARGET_COLUMN_MAP[target],
                "features_csv": features_csv,
                "labels_csv": labels_csv,
                "tree_manifest_source": str(tree_manifest) if tree_manifest.exists() else "-",
                "hpo_trials": tcn_cfg.get("hpo_trials"),
                "cv_splits": tcn_cfg.get("cv_splits"),
                "hpo_resume": bool(tcn_cfg.get("hpo_resume", True)),
                "hpo_storage": tcn_cfg.get("hpo_storage"),
                "hpo_adaptive_stop": bool(tcn_cfg.get("hpo_adaptive_stop", False)),
                "hpo_adaptive_min_completed_trials": tcn_cfg.get("hpo_adaptive_min_completed_trials"),
                "hpo_adaptive_no_improve_trials": tcn_cfg.get("hpo_adaptive_no_improve_trials"),
                "hpo_adaptive_min_delta": tcn_cfg.get("hpo_adaptive_min_delta"),
                "progress_snapshot": str(progress_json),
                "progress_events": str(progress_ndjson),
                "resume": bool(resume),
            },
            style="bright_magenta",
        )

        orchestrator = TrainOrchestrator(conf_dir=CONFIG_DIR, artifact_root=str(root))
        train_df = orchestrator.build_training_frame(
            asset=ASSET,
            tf_dir=tf_dir,
            features_csv=features_csv if Path(features_csv).exists() else None,
            labels_csv=labels_csv if Path(labels_csv).exists() else None,
            features_out=features_csv,
            labels_out=labels_csv,
            merged_out=str(merged_out),
            resume=resume,
            force_rebuild=False,
        )
        console_stage(
            "TCN training frame ready",
            f"rows={fmt_num(len(train_df))} path={merged_out}",
            status="ok",
        )

        trainer = TCNSpecialistTrainer(
            asset=ASSET,
            target=target,
            config=tcn_cfg,
            tree_manifest_path=tree_manifest if tree_manifest.exists() else None,
        )
        train_result = trainer.train(train_df)
        metrics = train_result["metrics"]
        model_obj = train_result["model"]
        feature_cols = train_result["feature_cols"]
        outcome = str(train_result.get("outcome", "trained"))

        registry_dir = _resolve_registry_dir(unified_cfg)
        versioner = ModelVersionManager(str(registry_dir / ".model_versions.json"))
        version = versioner.new_version(f"{ASSET}_tcn")
        manifest = _persist_tcn_result(
            asset=ASSET,
            target=target,
            registry_dir=registry_dir,
            version=version,
            train_df=train_df,
            tf_dir=tf_dir,
            features_csv=features_csv,
            labels_csv=labels_csv,
            merged_out=merged_out,
            model_state_out=model_state_out,
            manifest_out=manifest_out,
            feature_cols=feature_cols,
            metrics=metrics,
            model_obj=model_obj,
            outcome=outcome,
        )

        console_stage(
            "TCN model launcher complete",
            (
                f"model={target}_tcn version={version} outcome={outcome} "
                f"cv_score={metrics.get('cv_score'):.4f}"
            ),
            status="warn" if outcome == "checkpoint_saved" else "ok",
        )
        persisted = True
        return manifest
    finally:
        console_stage(
            "TCN training runtime",
            f"target={target} elapsed={fmt_seconds(time.perf_counter() - started)}",
            status="ok" if persisted else "warn",
        )


def deepcopy_dict(value: dict) -> dict:
    return {k: v for k, v in (value or {}).items()}


def _persist_tcn_result(
    *,
    asset: str,
    target: str,
    registry_dir: Path,
    version: str,
    train_df,
    tf_dir: str,
    features_csv: str,
    labels_csv: str,
    merged_out: Path,
    model_state_out: Path,
    manifest_out: Path,
    feature_cols: list,
    metrics: dict,
    model_obj,
    outcome: str,
) -> dict:
    registry = ModelRegistry(str(registry_dir))
    model_name_asset = f"{asset}_{target}_tcn"
    model_name_alias = f"{target}_tcn"
    model_config = {
        "features": feature_cols,
        "framework": "torch_tcn",
        "target": target,
        "label_col": TARGET_COLUMN_MAP[target],
        "training_outcome": outcome,
        "tcn": {
            "best_params": metrics.get("best_params", {}),
            "feature_transform_dim": metrics.get("feature_transform_dim"),
            "decision_threshold": (metrics.get("threshold_tuning", {}) or {}).get("threshold"),
        },
    }

    registry.save_model(model_name_asset, version, model_obj, None, model_config)
    registry.save_model(model_name_alias, version, model_obj, None, model_config)
    registry.save_metrics(model_name_asset, version, metrics)
    registry.save_metrics(model_name_alias, version, metrics)

    model_state = {
        "asset": asset,
        "version": version,
        "requested_models": [f"{target}_tcn"],
        "trained_models": [f"{target}_tcn"],
        "dependency_models": [],
        "registry_dir": str(registry_dir),
        "outcome": outcome,
        "checkpoint_interrupted": bool(metrics.get("checkpoint_interrupted", False)),
        "hpo_trials_completed": metrics.get("hpo_trials_completed"),
        "hpo_trials_requested": metrics.get("hpo_trials"),
    }
    save_json(model_state_out, model_state)

    manifest = {
        "asset": asset,
        "version": version,
        "rows": int(len(train_df)),
        "registry_dir": str(registry_dir),
        "tf_dir": tf_dir,
        "features_csv": features_csv,
        "labels_csv": labels_csv,
        "features_out": features_csv,
        "labels_out": labels_csv,
        "merged_out": str(merged_out),
        "model_state_out": str(model_state_out),
        "requested_models": [f"{target}_tcn"],
        "trained_models": [f"{target}_tcn"],
        "dependency_models": [],
        "outcome": outcome,
        "checkpoint_interrupted": bool(metrics.get("checkpoint_interrupted", False)),
        "metrics": {
            "summary": {
                "avg_cv_score": metrics.get("cv_score"),
                "specialists_with_metrics": [f"{target}_tcn"],
            },
            "by_model": {
                f"{target}_tcn": metrics,
            },
        },
        "governance": {
            "submitted": False,
            "reason": (
                "TCN checkpoint artifact saved from latest completed HPO trial; "
                "resume the same target to continue remaining trials."
                if outcome == "checkpoint_saved"
                else "TCN benchmark model trained for comparison; promotion workflow remains tree-stack based."
            ),
            "available_metrics": [f"{target}_tcn"],
            "acceptance_gate": (metrics.get("acceptance", {}) or {}).get("gate"),
            "stability_gate": ((metrics.get("stability", {}) or {}).get("gate")),
        },
    }
    save_json(manifest_out, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTCUSD specialist TCN model with Optuna HPO.")
    parser.add_argument(
        "--target",
        default="flow_1h",
        choices=sorted(TARGET_COLUMN_MAP.keys()),
        help="Specialist target to train (default: flow_1h).",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna HPO trial count.")
    parser.add_argument("--cv-splits", type=int, default=None, help="Override time-series CV split count.")
    parser.add_argument(
        "--adaptive-stop",
        action="store_true",
        help="Enable adaptive HPO early stop when no meaningful best-score improvement is observed.",
    )
    parser.add_argument(
        "--adaptive-min-completed-trials",
        type=int,
        default=None,
        help="Minimum completed trials before adaptive stop can trigger.",
    )
    parser.add_argument(
        "--adaptive-no-improve-trials",
        type=int,
        default=None,
        help="Stop after this many completed trials without improving best score by min delta.",
    )
    parser.add_argument(
        "--adaptive-min-delta",
        type=float,
        default=None,
        help="Minimum absolute score improvement treated as meaningful for adaptive stop.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force rebuild of merged training frame instead of using cache.",
    )
    args = parser.parse_args()
    run_target(
        args.target,
        trials=args.trials,
        cv_splits=args.cv_splits,
        adaptive_stop=True if args.adaptive_stop else None,
        adaptive_min_completed_trials=args.adaptive_min_completed_trials,
        adaptive_no_improve_trials=args.adaptive_no_improve_trials,
        adaptive_min_delta=args.adaptive_min_delta,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
