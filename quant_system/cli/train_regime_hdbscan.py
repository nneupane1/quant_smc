"""
Train HDBSCAN regime/cluster model on a given timeframe CSV.

Example:
    python -m quant_system.cli.train_regime_hdbscan ^
      --input data/features_xbtusd ^
      --asset XBTUSD ^
      --tf 15m ^
      --out-dir models/regime_hdbscan_15m
"""

import argparse
from pathlib import Path
import sys
import json
import pandas as pd

from quant_system.ml.regime.hdbscan_clustering import (
    HDBSCANConfig,
    HDBSCANClusterer,
    _HAS_HDBSCAN,
)


def main():
    ap = argparse.ArgumentParser(description="Train HDBSCAN regime/cluster model on a TF CSV.")
    ap.add_argument("--input", required=True, help="Input directory containing {asset}_{tf}.csv")
    ap.add_argument("--asset", required=True, help="Asset symbol prefix, e.g., XBTUSD")
    ap.add_argument("--tf", default="15m", help="Timeframe suffix used in filename (default: 15m)")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + meta")
    ap.add_argument("--min-cluster-size", type=int, default=None, help="Override min_cluster_size")
    ap.add_argument("--min-samples", type=int, default=None, help="Override min_samples")
    ap.add_argument("--cseps", type=float, default=None, help="Override cluster_selection_epsilon")
    ap.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of feature columns to use (optional; default is model config defaults).",
    )
    args = ap.parse_args()

    if not _HAS_HDBSCAN:
        print("[error] hdbscan is not installed. Please `pip install hdbscan` and retry.")
        sys.exit(1)

    inp_dir = Path(args.input)
    tf_file = inp_dir / f"{args.asset}_{args.tf}.csv"
    if not tf_file.exists():
        raise FileNotFoundError(f"Missing file: {tf_file}")

    df = pd.read_csv(tf_file, parse_dates=["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # Build config with overrides
    cfg = HDBSCANConfig()
    if args.min_cluster_size is not None:
        cfg.min_cluster_size = args.min_cluster_size
    if args.min_samples is not None:
        cfg.min_samples = args.min_samples
    if args.cseps is not None:
        cfg.cluster_selection_epsilon = args.cseps
    if args.features:
        cfg.feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    trainer = HDBSCANClusterer(cfg)
    report = trainer.fit(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(out_dir)

    with open(out_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[ok] trained HDBSCAN model -> {out_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
