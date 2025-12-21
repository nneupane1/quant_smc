"""
Train a NARX GBM return forecaster on 15m feature spine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_system.models.narx.narx_gbm import NARXGBM, NARXGBMConfig
from quant_system.models.narx.targets import add_forward_targets
from quant_system.utils.logger import get_logger

LOG = get_logger("train_return_forecaster")


def main():
    ap = argparse.ArgumentParser(description="Train NARX GBM return forecaster.")
    ap.add_argument("--in", dest="inp", required=True, help="Input 15m features CSV (must include dt, close).")
    ap.add_argument("--out-dir", required=True, help="Directory to save model artifacts.")
    ap.add_argument("--horizon", type=int, default=8, help="Horizon in bars for forward return (default 8).")
    ap.add_argument("--y-col", default=None, help="Optional explicit target column; else ret_fwd_{horizon}.")
    ap.add_argument("--exog", default=None, help="Comma-separated list of exogenous feature columns to include.")
    args = ap.parse_args()

    feat_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading features from %s", feat_path)
    df = pd.read_csv(feat_path, parse_dates=["dt"]).sort_values("dt").reset_index(drop=True)

    target_col = args.y_col
    if target_col is None or target_col not in df.columns:
        df = add_forward_targets(df, horizons=(args.horizon,))
        target_col = f"ret_fwd_{args.horizon}"
        LOG.info("Added forward target column: %s", target_col)

    exog_cols = None
    if args.exog:
        exog_cols = [c.strip() for c in args.exog.split(",") if c.strip()]

    cfg = NARXGBMConfig(
        horizon_bars=args.horizon,
        y_col=target_col,
        exog_cols=exog_cols,
    )

    model = NARXGBM(cfg)
    report = model.fit(df)
    model.save(str(out_dir))

    report_path = out_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    LOG.info("Saved model to %s", out_dir)
    LOG.info("Report: %s", report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
