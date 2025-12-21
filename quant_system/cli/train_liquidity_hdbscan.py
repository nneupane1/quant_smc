#!/usr/bin/env python3
"""
Train HDBSCAN-based liquidity/microstructure clustering on 15m or 1h features.

Example:
  python -m quant_system.cli.train_liquidity_hdbscan \
    --input data/features_xbtusd/XBTUSD_features.csv \
    --out models/liquidity_hdbscan_15m \
    --tf 15m
"""

import argparse
from pathlib import Path
import json

import pandas as pd

from quant_system.ml.regime.hdbscan_trainer import (
    HDBSCANConfig,
    LiquidityClusterTrainer,
)


def _parse_feature_cols(raw: str | None):
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Train HDBSCAN liquidity/microstructure clusters (15m or 1h)."
    )
    ap.add_argument("--input", required=True, help="Input features CSV with dt column")
    ap.add_argument("--out", required=True, help="Output directory for model artifacts")
    ap.add_argument(
        "--tf",
        default="15m",
        choices=["15m", "1h"],
        help="Timeframe to cluster on (default 15m)",
    )
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--cluster-selection-method", default="eom")
    ap.add_argument("--allow-single-cluster", action="store_true", help="Allow single cluster outputs")
    ap.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated feature columns to use (optional). If omitted, trainer defaults are used.",
    )
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.input, parse_dates=["dt"]).sort_values("dt").reset_index(drop=True)

    cfg = HDBSCANConfig(
        tf=args.tf,
        feature_cols=_parse_feature_cols(args.feature_cols),
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        epsilon=args.epsilon,
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
    )

    trainer = LiquidityClusterTrainer(cfg)
    states = trainer.fit(df, tf=args.tf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(out_dir, states)

    summary = {
        "n_clusters": int(trainer.n_clusters_),
        "noise_rate": float(trainer.noise_rate_),
        "tf": args.tf,
        "out": str(out_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
