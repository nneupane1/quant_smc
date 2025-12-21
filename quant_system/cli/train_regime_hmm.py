#!/usr/bin/env python3
"""
Train a Bayesian Gaussian HMM for regime detection on 6h/12h features.

Usage:
  python -m quant_system.cli.train_regime_hmm \
    --input data/features_xbtusd/XBTUSD_features.csv \
    --out-dir models/regime_hmm_xbtusd \
    --states 5 \
    --covariance full \
    --seed 42 \
    --features ret_6h,vol_6h,dispersion_6h,ret_12h,vol_12h,dispersion_12h
"""

import argparse
from pathlib import Path
import pandas as pd

from quant_system.models.regime.hmm_trainer import HMMConfig, train_hmm


def main():
    ap = argparse.ArgumentParser(description="Train Bayesian Gaussian HMM regime model (6h/12h).")
    ap.add_argument("--input", required=True, help="Input CSV with regime features (6h/12h).")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + meta + states CSV.")
    ap.add_argument("--states", type=int, default=5, help="Number of HMM states (default 5).")
    ap.add_argument("--covariance", default="full", help="Covariance type: full|diag|tied (default full).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    ap.add_argument("--features", default=None, help="Comma-separated list of feature columns to use.")
    ap.add_argument("--no-scale", action="store_true", help="Disable feature scaling before HMM.")
    args = ap.parse_args()

    feat_cols = None
    if args.features:
        feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    cfg = HMMConfig(
        n_states=args.states,
        covariance_type=args.covariance,
        random_state=args.seed,
        scale=not args.no_scale,
        feature_cols=feat_cols,
    )

    df = pd.read_csv(args.input, parse_dates=["dt"] if "dt" in pd.read_csv(args.input, nrows=0).columns else None)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    artifacts = train_hmm(df, cfg, args.out_dir)

    meta = artifacts.get("meta", {})
    states_path = Path(args.out_dir) / "regime_states.csv"

    print("[HMM] saved to", args.out_dir)
    print(f"[HMM] states={cfg.n_states} cov={cfg.covariance_type} scale={cfg.scale}")
    if meta:
        ll = meta.get("loglik_per_sample")
        rows = meta.get("n_rows")
        feats = meta.get("n_features")
        print(f"[HMM] loglik/sample={ll:.6f} rows={rows} features={feats}")

    if states_path.exists():
        try:
            sdf = pd.read_csv(states_path)
            counts = sdf["state"].value_counts().sort_index()
            print("[HMM] state counts:")
            for st, cnt in counts.items():
                print(f"  state {st}: {cnt}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
