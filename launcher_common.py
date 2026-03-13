"""
Shared helpers for top-level BTCUSD launch scripts.
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas",))

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

from quant_system.cli.common import load_or_build_features, load_or_build_labels, read_frame
from quant_system.config.config_loader import ConfigLoader
from quant_system.train_orchestrator import TrainOrchestrator
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_seconds

ASSET = "BTCUSD"
CONFIG_DIR = "quant_system/config"


def load_cfg():
    loader = ConfigLoader(CONFIG_DIR)
    return loader, loader.load()


def default_tf_dir(cfg: dict) -> str:
    return str(Path((cfg.get("paths", {}) or {}).get("tf", "data/tf")))


def default_features_csv() -> str:
    return f"artifacts/features/{ASSET}/{ASSET}_features.csv"


def default_labels_csv() -> str:
    return f"artifacts/labels/{ASSET}/{ASSET}_labels.csv"


def default_training_root(model_name: str) -> Path:
    return Path("artifacts/train") / ASSET / model_name


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_manifest_cv_score(manifest: Dict[str, Any], model_name: str) -> Optional[float]:
    metrics = (manifest.get("metrics", {}) or {}).get("by_model", {}) or {}
    metric_keys = [model_name]
    for key in metric_keys:
        row = metrics.get(key, {})
        score = row.get("cv_score") if isinstance(row, dict) else None
        if isinstance(score, (int, float)):
            return float(score)
    return None


def inspect_training_artifacts(model_name: str) -> Dict[str, Any]:
    root = default_training_root(model_name)
    manifest_path = root / "train_manifest.json"
    model_state_path = root / "model_state.json"
    merged_path = root / "training_frame.csv"
    manifest = _read_json(manifest_path)
    version = str(manifest.get("version") or "")
    cv_score = _extract_manifest_cv_score(manifest, model_name)
    return {
        "model": model_name,
        "artifact_root": str(root),
        "manifest_path": str(manifest_path),
        "model_state_path": str(model_state_path),
        "training_frame_path": str(merged_path),
        "manifest_exists": manifest_path.exists(),
        "model_state_exists": model_state_path.exists(),
        "training_frame_exists": merged_path.exists(),
        "manifest_mtime": manifest_path.stat().st_mtime if manifest_path.exists() else None,
        "version": version or "-",
        "cv_score": cv_score,
        "trained_models": manifest.get("trained_models", []) if isinstance(manifest, dict) else [],
        "dependency_models": manifest.get("dependency_models", []) if isinstance(manifest, dict) else [],
    }


def _is_up_to_date(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return False
    t_mtime = target.stat().st_mtime
    for src in sources:
        if src.exists() and src.stat().st_mtime > t_mtime:
            return False
    return True


def _elapsed_since(started_at: float) -> str:
    return fmt_seconds(time.perf_counter() - started_at)


def build_features(*, tf_dir: Optional[str] = None, features_out: Optional[str] = None, resume: bool = True):
    started_at = time.perf_counter()
    try:
        loader, cfg = load_cfg()
        tf_dir = tf_dir or default_tf_dir(cfg)
        features_out = features_out or default_features_csv()
        console_rule(f"Build Features | {ASSET}", style="cyan")
        console_kv(
            "Feature Plan",
            {
                "asset": ASSET,
                "tf_dir": tf_dir,
                "features_out": features_out,
                "resume": bool(resume),
            },
            style="cyan",
        )
        features_path = Path(features_out)
        tf_sources = [Path(tf_dir) / f"{ASSET}_{tf}.csv" for tf in ("15m", "1h", "6h", "12h")]
        if resume and _is_up_to_date(features_path, tf_sources):
            console_stage(
                "Features cache hit",
                f"path={features_out}",
                status="ok",
            )
            df = read_frame(str(features_path))
            return df, features_out

        df = load_or_build_features(
            loader,
            asset=ASSET,
            tf_dir=tf_dir,
            features_out=features_out,
        )
        console_stage(
            "Features ready",
            f"rows={len(df)} path={features_out}",
            status="ok",
        )
        return df, features_out
    finally:
        console_stage(
            "Build features runtime",
            f"elapsed={_elapsed_since(started_at)}",
            status="info",
        )


def build_labels(
    *,
    features_csv: Optional[str] = None,
    labels_out: Optional[str] = None,
    resume: bool = True,
):
    started_at = time.perf_counter()
    try:
        loader, _ = load_cfg()
        features_csv = features_csv or default_features_csv()
        labels_out = labels_out or default_labels_csv()
        console_rule(f"Build Labels | {ASSET}", style="yellow")
        console_kv(
            "Label Plan",
            {
                "asset": ASSET,
                "features_csv": features_csv,
                "labels_out": labels_out,
                "resume": bool(resume),
            },
            style="yellow",
        )
        labels_path = Path(labels_out)
        features_path = Path(features_csv) if features_csv else None
        active_profile_path = Path("artifacts/label_profiles/active_label_profile.json")
        if resume and labels_path.exists():
            sources = [s for s in (features_path, active_profile_path) if s is not None]
            if _is_up_to_date(labels_path, [s for s in sources if s is not None]):
                console_stage(
                    "Labels cache hit",
                    f"path={labels_out}",
                    status="ok",
                )
                labels_df = read_frame(str(labels_path))
                return labels_df, labels_out

        features_df = load_or_build_features(
            loader,
            asset=ASSET,
            features_csv=features_csv,
        )
        labels_df = load_or_build_labels(
            loader,
            features_df=features_df,
            labels_out=labels_out,
        )
        console_stage(
            "Labels ready",
            f"rows={len(labels_df)} path={labels_out}",
            status="ok",
        )
        return labels_df, labels_out
    finally:
        console_stage(
            "Build labels runtime",
            f"elapsed={_elapsed_since(started_at)}",
            status="info",
        )


def train_model(
    model_name: str,
    *,
    tf_dir: Optional[str] = None,
    features_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    resume: bool = True,
):
    started_at = time.perf_counter()
    try:
        _, cfg = load_cfg()
        tf_dir = tf_dir or default_tf_dir(cfg)
        features_csv = features_csv or default_features_csv()
        labels_csv = labels_csv or default_labels_csv()
        root = default_training_root(model_name)
        pre_status = inspect_training_artifacts(model_name)
        console_rule(f"Train Model | {ASSET} | {model_name}", style="green")
        console_kv(
            "Model Launch Plan",
            {
                "asset": ASSET,
                "model": model_name,
                "artifact_root": root,
                "manifest": pre_status["manifest_path"],
                "training_frame": pre_status["training_frame_path"],
                "features_csv": features_csv,
                "labels_csv": labels_csv,
                "resume": bool(resume),
                "existing_version": pre_status["version"],
                "existing_cv_score": (
                    f"{float(pre_status['cv_score']):.4f}"
                    if isinstance(pre_status.get("cv_score"), (int, float))
                    else "-"
                ),
            },
            style="green",
        )
        orchestrator = TrainOrchestrator(conf_dir=CONFIG_DIR, artifact_root=str(root))
        manifest = orchestrator.run_asset(
            asset=ASSET,
            tf_dir=tf_dir,
            features_csv=features_csv if Path(features_csv).exists() else None,
            labels_csv=labels_csv if Path(labels_csv).exists() else None,
            features_out=features_csv,
            labels_out=labels_csv,
            merged_out=str(root / "training_frame.csv"),
            manifest_out=str(root / "train_manifest.json"),
            model_state_out=str(root / "model_state.json"),
            models=[model_name],
            resume=resume,
        )
        post_status = inspect_training_artifacts(model_name)
        outcome = "trained"
        if (
            resume
            and bool(pre_status.get("manifest_exists"))
            and bool(post_status.get("manifest_exists"))
            and pre_status.get("manifest_mtime") == post_status.get("manifest_mtime")
            and pre_status.get("version") == post_status.get("version")
        ):
            outcome = "resume_hit"
        console_stage(
            "Model launcher complete",
            f"model={model_name} version={manifest['version']} outcome={outcome}",
            status="ok",
        )
        console_kv(
            "Model Launch Summary",
            {
                "model": model_name,
                "outcome": outcome,
                "version": post_status["version"],
                "cv_score": (
                    f"{float(post_status['cv_score']):.4f}"
                    if isinstance(post_status.get("cv_score"), (int, float))
                    else "-"
                ),
                "trained_models": ", ".join(post_status.get("trained_models", [])) or "-",
                "dependency_models": ", ".join(post_status.get("dependency_models", [])) or "-",
                "manifest": post_status["manifest_path"],
            },
            style="green",
        )
        return manifest
    finally:
        console_stage(
            "Training runtime",
            f"model={model_name} elapsed={_elapsed_since(started_at)}",
            status="info",
        )
