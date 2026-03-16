"""
CLI entrypoint for historical backtesting.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

from quant_system.backtest.core.backtester import Backtester
from quant_system.backtest.policy_dataset import build_trade_policy_dataset
from quant_system.backtest.report_generator import generate_backtest_artifacts
from quant_system.cli.common import (
    default_asset,
    default_conf_dir,
    default_dashboard_path,
    load_or_build_features,
    load_registry,
    resolve_conf_dir,
    save_json,
)
from quant_system.config.config_loader import ConfigLoader
from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter
from quant_system.telemetry.runtime import start_terminal_server
from quant_system.utils.logger import get_logger, runtime_logged

LOG = get_logger("backtest_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Run backtest engine from a feature CSV or TF directory.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Asset symbol, e.g. XBTUSD")
    parser.add_argument("--features", default=None, help="15m feature CSV to backtest.")
    parser.add_argument("--tf-dir", default=None, help="Directory containing {ASSET}_{15m,1h,6h,12h}.csv.")
    parser.add_argument("--features-out", default=None, help="Optional path to persist built features CSV.")
    parser.add_argument("--model-registry", default=None, help="Model registry path.")
    parser.add_argument("--out-dir", default="artifacts/backtest/latest", help="Directory for backtest artifacts.")
    parser.add_argument("--start-date", default=None, help="Inclusive backtest start date, e.g. 2017-01-01.")
    parser.add_argument("--end-date", default=None, help="Inclusive backtest end date, e.g. 2026-03-16.")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after artifacts are written.")
    parser.add_argument("--terminal-server", action="store_true", help="Start FastAPI + WebSocket terminal backend in-process.")
    parser.add_argument("--terminal-host", default="127.0.0.1", help="Host for in-process terminal backend.")
    parser.add_argument("--terminal-port", default=8100, type=int, help="Port for in-process terminal backend.")
    return parser.parse_args()


@runtime_logged("Backtest CLI runtime")
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
    if "dt" not in features_df.columns:
        raise KeyError("Backtest features must include a 'dt' column for date-window filtering.")
    features_df = features_df.copy()
    features_df["dt"] = pd.to_datetime(features_df["dt"], utc=True, errors="coerce")
    features_df = features_df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    default_start = ((cfg.get("data", {}) or {}).get("start_date") or "2017-01-01")
    start_raw = args.start_date or default_start
    end_raw = args.end_date
    start_dt = pd.to_datetime(start_raw, utc=True, errors="coerce") if start_raw else None
    end_dt = pd.to_datetime(end_raw, utc=True, errors="coerce") if end_raw else None
    if start_dt is not None:
        features_df = features_df.loc[features_df["dt"] >= start_dt]
    if end_dt is not None:
        features_df = features_df.loc[features_df["dt"] <= end_dt]
    features_df = features_df.reset_index(drop=True)
    if features_df.empty:
        raise ValueError(
            f"Backtest feature frame is empty after date filtering start={start_raw!r} end={end_raw!r}."
        )
    LOG.info(
        "Backtest window | asset=%s start=%s end=%s rows=%s",
        asset,
        features_df["dt"].min(),
        features_df["dt"].max(),
        len(features_df),
    )

    registry = load_registry(cfg, args.model_registry)
    dashboard_adapter = ForwardDashboardAdapter()
    if args.terminal_server:
        handle = start_terminal_server(args.terminal_host, args.terminal_port)
        LOG.info("Terminal backend available at %s (ws: %s)", handle.http_url, handle.ws_url)

    engine = Backtester(loader, registry, dashboard_adapter=dashboard_adapter)
    result = engine.run({asset: features_df})

    out_dir = Path(args.out_dir)
    paths = generate_backtest_artifacts(
        result,
        out_dir,
        candles=features_df,
        smc_features=features_df,
        starting_equity=float(cfg.get("execution", {}).get("starting_equity", 0.0)),
    )
    dataset_path = out_dir / "trade_policy_dataset.csv"
    dataset_meta_path = out_dir / "trade_policy_dataset_meta.json"
    build_trade_policy_dataset(
        out_dir,
        output_csv=dataset_path,
        output_meta=dataset_meta_path,
        config=((loader.load_yaml("models.yaml") or {}).get("trade_policy_training", {}) or {}).get("dataset", {}),
    )
    save_json(
        out_dir / "run_manifest.json",
        {
            "asset": asset,
            "window": {
                "start": features_df["dt"].min(),
                "end": features_df["dt"].max(),
                "rows": len(features_df),
            },
            "artifacts": {
                **paths,
                "trade_policy_dataset": dataset_path,
                "trade_policy_dataset_meta": dataset_meta_path,
            },
            "metrics": result.get("metrics", {}),
        },
    )

    LOG.info("Backtest completed for %s. Artifacts written to %s", asset, out_dir)

    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", default_dashboard_path(__file__)], check=False)


if __name__ == "__main__":
    main()
