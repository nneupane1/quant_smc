"""
CLI entrypoint for feature -> label -> model training.
"""

import argparse
from pathlib import Path

import pandas as pd

from quant_system.cli.common import (
    default_asset,
    default_conf_dir,
    load_or_build_features,
    load_or_build_labels,
    load_registry,
    resolve_conf_dir,
    save_json,
)
from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.training.model_trainer import ModelTrainer
from quant_system.utils.logger import get_logger

LOG = get_logger("train_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Train models from a TF directory or prepared feature/label CSVs.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Asset symbol, e.g. XBTUSD")
    parser.add_argument("--tf-dir", default=None, help="Directory containing {ASSET}_{15m,1h,6h,12h}.csv.")
    parser.add_argument("--features", default=None, help="Prepared features CSV.")
    parser.add_argument("--labels", default=None, help="Prepared labels CSV or full feature+label CSV.")
    parser.add_argument("--features-out", default=None, help="Optional output CSV for built features.")
    parser.add_argument("--labels-out", default=None, help="Optional output CSV for generated labels.")
    parser.add_argument("--models", default=None, help="Comma-separated model list, e.g. liq_flow,flow_1h,meta_model.")
    parser.add_argument("--model-registry", default=None, help="Model registry output directory.")
    parser.add_argument("--merged-out", default="artifacts/train/latest/training_frame.csv", help="Persist merged training frame.")
    return parser.parse_args()


def _merge_training_frame(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df is features_df:
        return labels_df
    if "dt" in features_df.columns and "dt" in labels_df.columns:
        drop_cols = [c for c in labels_df.columns if c in features_df.columns and c != "dt"]
        return features_df.merge(labels_df.drop(columns=drop_cols), on="dt", how="inner")
    return labels_df


def main():
    args = parse_args()
    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info("Loading configuration from %s ...", conf_dir)
    loader = ConfigLoader(conf_dir)
    cfg = loader.load()
    asset = default_asset(cfg, args.asset)

    features_df = load_or_build_features(
        loader,
        asset=asset,
        features_csv=args.features,
        tf_dir=args.tf_dir,
        features_out=args.features_out,
    )
    labels_df = load_or_build_labels(
        loader,
        features_df=features_df,
        labels_csv=args.labels,
        labels_out=args.labels_out,
    )
    train_df = _merge_training_frame(features_df, labels_df)

    merged_out = Path(args.merged_out)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(merged_out, index=False)

    registry = load_registry(cfg, args.model_registry)
    trainer = ModelTrainer(loader, registry)
    train_result = trainer.train_asset_bundle(
        train_df,
        asset,
        selected_models=[m.strip() for m in args.models.split(",")] if args.models else None,
    )
    version = train_result["version"]

    manifest = {
        "asset": asset,
        "version": version,
        "rows": int(len(train_df)),
        "requested_models": train_result.get("requested_models", []),
        "trained_models": train_result.get("trained_models", []),
        "features_out": args.features_out,
        "labels_out": args.labels_out,
        "merged_out": str(merged_out),
        "registry_dir": registry.base_dir,
    }
    save_json(merged_out.parent / "train_manifest.json", manifest)
    LOG.info("Training complete for %s version=%s", asset, version)


if __name__ == "__main__":
    main()
