"""
Train an unsupervised Gaussian HMM for regimes on a 6h/12h feature set.

Example:
    python -m quant_system.cli.train_hmm \
      --input data/features_xbtusd/XBTUSD_features.csv \
      --features ret_1h,ret_6h,ret_vol_6h,atr_6h,vol_pct_12h \
      --n-states 5 \
      --out models/hmm_regime
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _parse_features(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [c.strip() for c in arg.split(",") if c.strip()]


def main():
    ap = argparse.ArgumentParser(description="Train an unsupervised Gaussian HMM for regimes.")
    ap.add_argument("--input", required=True, help="CSV with regime features (dt + columns).")
    ap.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. If omitted, will try a sensible default set.",
    )
    ap.add_argument("--n-states", type=int, default=5, help="Number of HMM states (4-6 typical).")
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory to store model.joblib and meta.json",
    )
    args = ap.parse_args()
    try:
        from quant_system.models.regime.hmm_trainer import HMMConfig, HMMTrainer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(f"hmm trainer unavailable: {exc}") from exc

    df = pd.read_csv(args.input, parse_dates=["dt"], low_memory=False)
    feat_list = _parse_features(args.features)
    if not feat_list:
        # fall back to common regime features if present
        candidates = [
            "ret_1h",
            "ret_6h",
            "ret_12h",
            "ret_vol_6h",
            "ret_vol_12h",
            "atr_6h",
            "atr_12h",
            "vol_pct_6h",
            "vol_pct_12h",
        ]
        feat_list = [c for c in candidates if c in df.columns]
        if not feat_list:
            raise ValueError("No features specified and no default regime columns found in the CSV.")

    cfg = HMMConfig(n_states=args.n_states)
    trainer = HMMTrainer(cfg, feat_list)
    report = trainer.fit(df)

    out_dir = Path(args.out)
    trainer.save(out_dir)

    with (out_dir / "train_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[ok] HMM trained. States={cfg.n_states}  rows={len(df)}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
