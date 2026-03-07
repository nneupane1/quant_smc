"""
CLI entrypoint for offline forward-testing over prepared 15m feature rows.
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
from quant_system.forward_test.forward_engine import ForwardEngine
from quant_system.telemetry.runtime import start_terminal_server
from quant_system.utils.logger import get_logger, runtime_logged

LOG = get_logger("forward_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="Run forward-test engine over a feature CSV or TF directory.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Asset symbol, e.g. XBTUSD")
    parser.add_argument("--features", default=None, help="Prepared 15m feature CSV.")
    parser.add_argument("--tf-dir", default=None, help="Directory containing {ASSET}_{15m,1h,6h,12h}.csv.")
    parser.add_argument("--features-out", default=None, help="Optional output CSV for built features.")
    parser.add_argument("--model-registry", default=None, help="Model registry path.")
    parser.add_argument("--out-dir", default="forward_outputs", help="Output directory for forward artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on bars processed.")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after artifacts are written.")
    parser.add_argument("--terminal-server", action="store_true", help="Start FastAPI + WebSocket terminal backend in-process.")
    parser.add_argument("--terminal-host", default="127.0.0.1", help="Host for in-process terminal backend.")
    parser.add_argument("--terminal-port", default=8100, type=int, help="Port for in-process terminal backend.")
    return parser.parse_args()


@runtime_logged("Forward CLI runtime")
def main():
    os.environ.setdefault("QUANT_RUNTIME_LOGS", "0")
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
    ).sort_values("dt").reset_index(drop=True)
    if args.limit is not None:
        features_df = features_df.head(args.limit).copy()

    registry = load_registry(cfg, args.model_registry)
    adapter = ForwardDashboardAdapter()
    if args.terminal_server:
        handle = start_terminal_server(args.terminal_host, args.terminal_port)
        LOG.info("Terminal backend available at %s (ws: %s)", handle.http_url, handle.ws_url)
    engine = ForwardEngine(loader, registry, dashboard_adapter=adapter)
    engine.load_models("latest")

    for _, row in features_df.iterrows():
        engine.on_bar(asset, row.to_dict())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter.live_equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    pd.DataFrame(adapter.event_log).to_csv(out_dir / "events.csv", index=False)
    features_df.to_csv(out_dir / "bars.csv", index=False)
    save_json(out_dir / "snapshot.json", adapter.get_snapshot())
    pd.DataFrame(engine.closed_trades.values()).to_csv(out_dir / "closed_trades.csv", index=False)
    save_json(out_dir / "state.json", engine.state_snapshot())
    save_json(
        out_dir / "forward_manifest.json",
        {
            "asset": asset,
            "rows_processed": int(len(features_df)),
            "registry_dir": registry.base_dir,
            "out_dir": str(out_dir),
        },
    )

    LOG.info("Forward test completed for %s. Artifacts written to %s", asset, out_dir)

    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", default_dashboard_path(__file__)], check=False)


if __name__ == "__main__":
    main()
