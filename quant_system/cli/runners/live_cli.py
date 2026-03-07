"""
CLI entrypoint for live orchestration.

Default mode is safe dry-run over prepared bars/features.
Use `--stream` to poll Kraken 1m data and aggregate live.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

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
from quant_system.live.live_orchestrator import LiveOrchestrator
from quant_system.telemetry.runtime import start_terminal_server
from quant_system.utils.logger import get_logger, runtime_logged

LOG = get_logger("live_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Run live orchestration in stream or dry-run row mode.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Asset symbol, e.g. BTCUSD")
    parser.add_argument("--features", default=None, help="Prepared 15m feature CSV for dry-run mode.")
    parser.add_argument("--tf-dir", default=None, help="Directory containing {ASSET}_{15m,1h,6h,12h}.csv for dry-run mode.")
    parser.add_argument("--features-out", default=None, help="Optional output path for built features.")
    parser.add_argument("--model-registry", default=None, help="Model registry path.")
    parser.add_argument("--out-dir", default="live_outputs", help="Output directory for live artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on dry-run rows processed.")
    parser.add_argument("--stream", action="store_true", help="Use Kraken 1m polling stream instead of dry-run rows.")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after artifacts are written.")
    parser.add_argument("--terminal-server", action="store_true", help="Start FastAPI + WebSocket terminal backend in-process.")
    parser.add_argument("--terminal-host", default="127.0.0.1", help="Host for in-process terminal backend.")
    parser.add_argument("--terminal-port", default=8100, type=int, help="Port for in-process terminal backend.")
    return parser.parse_args()


@runtime_logged("Live CLI runtime")
def main():
    os.environ.setdefault("QUANT_RUNTIME_LOGS", "0")
    args = parse_args()
    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info("Loading configuration from %s ...", conf_dir)
    loader = ConfigLoader(conf_dir)
    cfg = loader.load()
    asset = default_asset(cfg, args.asset)

    registry = load_registry(cfg, args.model_registry)
    adapter = ForwardDashboardAdapter()
    if args.terminal_server:
        handle = start_terminal_server(args.terminal_host, args.terminal_port)
        LOG.info("Terminal backend available at %s (ws: %s)", handle.http_url, handle.ws_url)
    engine = LiveOrchestrator(loader, registry, dashboard_adapter=adapter)
    engine.load_models()

    processed_rows = 0
    bars_df = pd.DataFrame()
    if args.stream:
        engine.run()
    else:
        bars_df = load_or_build_features(
            loader,
            asset=asset,
            features_csv=args.features,
            tf_dir=args.tf_dir,
            features_out=args.features_out,
        ).sort_values("dt").reset_index(drop=True)
        if args.limit is not None:
            bars_df = bars_df.head(args.limit).copy()
        processed_rows = int(len(bars_df))
        engine.run_rows(asset, bars_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bars_df.empty:
        bars_df.to_csv(out_dir / "bars.csv", index=False)
    adapter.live_equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    pd.DataFrame(adapter.event_log).to_csv(out_dir / "events.csv", index=False)
    save_json(out_dir / "snapshot.json", adapter.get_snapshot())
    pd.DataFrame(engine.closed_trades.values()).to_csv(out_dir / "closed_trades.csv", index=False)
    save_json(out_dir / "state.json", engine.state_snapshot())
    save_json(
        out_dir / "live_manifest.json",
        {
            "asset": asset,
            "mode": "stream" if args.stream else "dry_run",
            "rows_processed": processed_rows,
            "registry_dir": registry.base_dir,
            "out_dir": str(out_dir),
            "live_enabled": bool(cfg.get("live_trading", {}).get("enabled", False)),
        },
    )

    LOG.info("Live run completed for %s. Artifacts written to %s", asset, out_dir)

    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", default_dashboard_path(__file__)], check=False)


if __name__ == "__main__":
    main()
