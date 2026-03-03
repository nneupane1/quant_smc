"""
CLI entrypoint for historical backtesting.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from quant_system.backtest.core.backtester import Backtester
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
from quant_system.utils.logger import get_logger

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
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after artifacts are written.")
    parser.add_argument("--terminal-server", action="store_true", help="Start FastAPI + WebSocket terminal backend in-process.")
    parser.add_argument("--terminal-host", default="127.0.0.1", help="Host for in-process terminal backend.")
    parser.add_argument("--terminal-port", default=8100, type=int, help="Port for in-process terminal backend.")
    return parser.parse_args()


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
    save_json(out_dir / "run_manifest.json", {"asset": asset, "artifacts": paths, "metrics": result.get("metrics", {})})

    LOG.info("Backtest completed for %s. Artifacts written to %s", asset, out_dir)

    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", default_dashboard_path(__file__)], check=False)


if __name__ == "__main__":
    main()
