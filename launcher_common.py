"""
Shared helpers for top-level BTCUSD launch scripts.
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas",))

from pathlib import Path
from typing import Optional

from quant_system.cli.common import load_or_build_features, load_or_build_labels
from quant_system.config.config_loader import ConfigLoader
from quant_system.train_orchestrator import TrainOrchestrator
from quant_system.utils.logger import console_kv, console_rule, console_stage

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


def build_features(*, tf_dir: Optional[str] = None, features_out: Optional[str] = None):
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
        },
        style="cyan",
    )
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


def build_labels(
    *,
    features_csv: Optional[str] = None,
    labels_out: Optional[str] = None,
):
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
        },
        style="yellow",
    )
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


def train_model(
    model_name: str,
    *,
    tf_dir: Optional[str] = None,
    features_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
):
    _, cfg = load_cfg()
    tf_dir = tf_dir or default_tf_dir(cfg)
    features_csv = features_csv or default_features_csv()
    labels_csv = labels_csv or default_labels_csv()
    root = default_training_root(model_name)
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
    )
    console_stage(
        "Model launcher complete",
        f"model={model_name} version={manifest['version']}",
        status="ok",
    )
    return manifest
