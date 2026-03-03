#!/usr/bin/env python3
"""
Train a higher-timeframe regime HMM on 6h/12h bars or feature rows.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Train Gaussian HMM regime model (6h/12h).")
    ap.add_argument("--input", required=True, help="Input CSV with dt and either OHLCV or regime feature columns.")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + scaler + meta + states CSV.")
    ap.add_argument("--states", type=int, default=5, help="Number of HMM states.")
    ap.add_argument("--covariance", default="diag", help="Covariance type: full|diag|tied|spherical.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--features", default=None, help="Optional comma-separated feature columns.")
    args = ap.parse_args()

    try:
        from quant_system.ml.training.regime_hmm_trainer import RegimeHMMConfig, RegimeHMMTrainer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(f"regime hmm trainer unavailable: {exc}") from exc

    feat_cols = None
    if args.features:
        feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    cfg = RegimeHMMConfig(
        n_states=args.states,
        covariance_type=args.covariance,
        seed=args.seed,
        feature_cols=feat_cols,
    )

    df = pd.read_csv(args.input, parse_dates=["dt"] if "dt" in pd.read_csv(args.input, nrows=0).columns else None)
    trainer = RegimeHMMTrainer(cfg)
    states = trainer.fit_transform(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_dir))
    states.to_csv(out_dir / "regime_states.csv", index=False)

    report = {
        "rows": int(len(states)),
        "states": int(cfg.n_states),
        "features": trainer.features_,
        "out_dir": str(out_dir),
    }
    with (out_dir / "train_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
