"""
Train an HDBSCAN liquidity/microstructure clustering model.

Usage (PowerShell example):
  python -m quant_system.cli.train_hdbscan `
    --input data/features_xbtusd/XBTUSD_features.csv `
    --out-dir models/hdbscan_liquidity `
    --features atr,volume,dollar_volume,vol_z,range_pct
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from quant_system.models.liquidity.hdbscan_trainer import (
    HDBSCANConfig,
    HDBSCANTrainer,
)


def _parse_features(val: Optional[str]) -> Optional[List[str]]:
    if not val:
        return None
    return [c.strip() for c in val.split(",") if c.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Train HDBSCAN liquidity-state model (unsupervised)."
    )
    ap.add_argument("--input", required=True, help="Input CSV with features.")
    ap.add_argument("--out-dir", required=True, help="Output directory for model.")
    ap.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of feature columns (default: auto).",
    )
    ap.add_argument("--min-cluster-size", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--cluster-selection-method", default="eom")
    ap.add_argument(
        "--allow-single-cluster",
        action="store_true",
        help="Allow single-cluster result.",
    )
    ap.add_argument(
        "--emit-labels",
        action="store_true",
        help="Persist labels.csv alongside the model.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["dt"]) if Path(args.input).exists() else pd.read_csv(args.input)

    cfg = HDBSCANConfig(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
        features=_parse_features(args.features),
        emit_labels=args.emit_labels,
    )

    trainer = HDBSCANTrainer(cfg)
    trainer.fit(df)
    trainer.save(args.out_dir)

    meta_path = Path(args.out_dir) / "meta.json"
    print(f"[ok] HDBSCAN model saved to {args.out_dir}")
    print(f"[meta] {meta_path}")
    if args.emit_labels:
        print(f"[labels] {Path(args.out_dir) / 'labels.csv'}")


if __name__ == "__main__":
    main()
