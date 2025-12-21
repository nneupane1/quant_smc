"""
Simple training orchestrator CLI.
Loads features + labels CSVs, merges them, and runs ModelTrainer for one asset.

Usage (PowerShell example):
  python -m quant_system.cli.train_models `
    --asset XBTUSD `
    --features data/features_xbtusd/XBTUSD_features.csv `
    --labels data/labels_xbtusd/XBTUSD_labels.csv `
    --models-out models
"""

import argparse
import os
import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.training.model_trainer import ModelTrainer
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.utils.logger import get_logger

LOG = get_logger("train_models_cli")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all models for one asset from feature/label CSVs.")
    p.add_argument("--config", default="quant_system/config", help="Config directory")
    p.add_argument("--asset", required=True, help="Asset symbol (e.g., XBTUSD)")
    p.add_argument("--features", required=True, help="Path to features CSV")
    p.add_argument("--labels", required=True, help="Path to labels CSV")
    p.add_argument("--models-out", default="models", help="Directory to write trained models/versions")
    return p.parse_args()


def load_data(feat_path: str, lbl_path: str) -> pd.DataFrame:
    LOG.info(f"Loading features from {feat_path}")
    df_feat = pd.read_csv(feat_path)
    LOG.info(f"Loading labels from {lbl_path}")
    df_lbl = pd.read_csv(lbl_path)

    if "dt" in df_feat.columns and "dt" in df_lbl.columns:
        df = df_feat.merge(df_lbl, on="dt", how="inner")
    else:
        df = pd.concat([df_feat, df_lbl], axis=1)
    LOG.info(f"Merged rows={len(df)} cols={len(df.columns)}")
    return df


def main():
    args = parse_args()
    cfg = ConfigLoader(args.config)
    registry = ModelRegistry(args.models_out)

    df = load_data(args.features, args.labels)

    trainer = ModelTrainer(cfg, registry)
    version = trainer.train_asset(df, args.asset)
    LOG.info(f"Training complete asset={args.asset} version={version} saved to {args.models_out}")


if __name__ == "__main__":
    main()
