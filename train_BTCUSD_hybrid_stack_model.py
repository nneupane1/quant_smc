from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "numpy", "sklearn"))

import argparse
import time
from pathlib import Path

from quant_system.cli.common import save_json
from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.registry.versioning import ModelVersionManager
from quant_system.ml.training.hybrid_stack_trainer import HYBRID_STACK_TARGETS, HybridStackTrainer
from quant_system.train_orchestrator import TrainOrchestrator
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_num, fmt_seconds

ASSET = "BTCUSD"
CONFIG_DIR = "quant_system/config"


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


def _persist_hybrid_stack_result(
    *,
    asset: str,
    target_model_name: str,
    dependency_models: list,
    route_slot: str,
    allow_unpinned_fallback: bool,
    registry_dir: Path,
    version: str,
    train_df,
    tf_dir: str,
    features_csv: str,
    labels_csv: str,
    merged_out: Path,
    model_state_out: Path,
    manifest_out: Path,
    metrics: dict,
    model_obj,
    model_cfg: dict,
    outcome: str,
) -> dict:
    registry = ModelRegistry(str(registry_dir))
    model_name_asset = f"{asset}_{target_model_name}"
    model_name_alias = target_model_name

    cfg_payload = dict(model_cfg or {})
    cfg_payload["training_outcome"] = outcome
    registry.save_model(model_name_asset, version, model_obj, None, cfg_payload)
    registry.save_model(model_name_alias, version, model_obj, None, cfg_payload)
    registry.save_metrics(model_name_asset, version, metrics)
    registry.save_metrics(model_name_alias, version, metrics)

    model_state = {
        "asset": asset,
        "version": version,
        "requested_models": [target_model_name],
        "trained_models": [target_model_name],
        "dependency_models": dependency_models,
        "route_slot": route_slot,
        "allow_unpinned_fallback": bool(allow_unpinned_fallback),
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
        "requested_models": [target_model_name],
        "trained_models": [target_model_name],
        "dependency_models": dependency_models,
        "route_slot": route_slot,
        "outcome": outcome,
        "checkpoint_interrupted": bool(metrics.get("checkpoint_interrupted", False)),
        "metrics": {
            "summary": {
                "avg_cv_score": metrics.get("cv_score"),
                "models_with_metrics": [target_model_name],
                "specialists_with_metrics": [],
            },
            "by_model": {
                target_model_name: metrics,
            },
        },
        "governance": {
            "submitted": False,
            "reason": (
                "Hybrid stack checkpoint artifact saved from latest completed challenger state; "
                "rerun the same target to continue remaining HPO."
                if outcome == "checkpoint_saved"
                else "Hybrid stack model trained from explicit mixed specialist routes; promotion remains explicit."
            ),
            "available_metrics": [target_model_name],
        },
    }
    save_json(manifest_out, manifest)
    return manifest


def run_target(
    target: str,
    *,
    slot: str = "hybrid_candidate",
    resume: bool = True,
    allow_unpinned_fallback: bool = False,
) -> dict:
    if target not in HYBRID_STACK_TARGETS:
        raise SystemExit(
            f"Unsupported hybrid stack target '{target}'. "
            f"Choose from: {', '.join(sorted(HYBRID_STACK_TARGETS))}"
        )

    started = time.perf_counter()
    persisted = False
    target_model_name = f"{target}_hybrid"
    root = Path("artifacts/train") / ASSET / target_model_name
    root.mkdir(parents=True, exist_ok=True)

    try:
        loader = ConfigLoader(CONFIG_DIR)
        unified_cfg = loader.load()
        tf_dir = _default_tf_dir(unified_cfg)
        features_csv = _default_features_csv()
        labels_csv = _default_labels_csv()
        merged_out = root / "training_frame.csv"
        model_state_out = root / "model_state.json"
        manifest_out = root / "train_manifest.json"
        registry_dir = _resolve_registry_dir(unified_cfg)
        registry = ModelRegistry(str(registry_dir))
        versioner = ModelVersionManager(str(registry_dir / ".model_versions.json"))

        console_rule(f"Train Hybrid Stack | {ASSET} | {target}", style="bright_cyan")
        console_kv(
            "Hybrid Stack Launch Plan",
            {
                "asset": ASSET,
                "target": target_model_name,
                "route_slot": slot,
                "allow_unpinned_fallback": bool(allow_unpinned_fallback),
                "features_csv": features_csv,
                "labels_csv": labels_csv,
                "artifact_root": str(root),
                "manifest": str(manifest_out),
                "resume": bool(resume),
            },
            style="bright_cyan",
        )

        if resume and manifest_out.exists():
            try:
                existing_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
                if (
                    str(existing_manifest.get("version") or "").strip()
                    and str(existing_manifest.get("outcome") or "trained") != "checkpoint_saved"
                ):
                    console_stage(
                        "Hybrid stack resume hit",
                        f"version={existing_manifest.get('version')} manifest={manifest_out}",
                        status="ok",
                    )
                    persisted = True
                    return existing_manifest
            except Exception:
                pass

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
            "Hybrid stack training frame ready",
            f"rows={fmt_num(len(train_df))} path={merged_out}",
            status="ok",
        )

        trainer = HybridStackTrainer(
            asset=ASSET,
            target=target,
            config_loader=loader,
            registry=registry,
            active_slot=slot,
            allow_unpinned_fallback=allow_unpinned_fallback,
            artifact_root=root,
        )
        train_result = trainer.train(train_df)
        metrics = train_result["metrics"]
        model_obj = train_result["model"]
        model_cfg = train_result["config"]
        outcome = str(train_result.get("outcome", "trained"))

        version = versioner.new_version(f"{ASSET}_hybrid_stack")
        dependency_models = list(trainer.required_specialists())
        manifest = _persist_hybrid_stack_result(
            asset=ASSET,
            target_model_name=target_model_name,
            dependency_models=dependency_models,
            route_slot=slot,
            allow_unpinned_fallback=allow_unpinned_fallback,
            registry_dir=registry_dir,
            version=version,
            train_df=train_df,
            tf_dir=tf_dir,
            features_csv=features_csv,
            labels_csv=labels_csv,
            merged_out=merged_out,
            model_state_out=model_state_out,
            manifest_out=manifest_out,
            metrics=metrics,
            model_obj=model_obj,
            model_cfg=model_cfg,
            outcome=outcome,
        )

        console_stage(
            "Hybrid stack launcher complete",
            (
                f"model={target_model_name} version={version} outcome={outcome} "
                f"cv_score={metrics.get('cv_score'):.4f} "
                f"delta_vs_tree={metrics.get('delta_vs_tree_cv_score')} "
                f"slot={slot}"
            ),
            status="warn" if outcome == "checkpoint_saved" else "ok",
        )
        persisted = True
        return manifest
    finally:
        console_stage(
            "Hybrid stack runtime",
            f"target={target_model_name} elapsed={fmt_seconds(time.perf_counter() - started)}",
            status="ok" if persisted else "warn",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTCUSD hybrid stack model over explicitly routed specialist winners.")
    parser.add_argument(
        "--target",
        default="meta_model",
        choices=sorted(HYBRID_STACK_TARGETS.keys()),
        help="Hybrid stack target to train.",
    )
    parser.add_argument(
        "--slot",
        default="hybrid_candidate",
        help="Active route slot containing explicit specialist winner routes.",
    )
    parser.add_argument(
        "--allow-unpinned-fallback",
        action="store_true",
        help="Allow unresolved specialists to fall back to best available bundle. Default is strict explicit routing.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force a fresh training run even if a manifest already exists.",
    )
    args = parser.parse_args()
    run_target(
        args.target,
        slot=args.slot,
        resume=not args.no_resume,
        allow_unpinned_fallback=args.allow_unpinned_fallback,
    )


if __name__ == "__main__":
    main()
