"""
CLI wrapper for backtesting.

Note: The full data-loading pipeline is not wired in this CLI yet.
Use this as a thin orchestrator once asset_frames provisioning is implemented.
"""

import argparse
import os
import sys
import subprocess

from quant_system.backtest.backtester import Backtester
from quant_system.ml.model_registry import ModelRegistry
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("backtest_cli")


def resolve_conf_dir(path: str) -> str:
    """
    Accept either a directory or a file inside the config directory.
    """
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def main():
    parser = argparse.ArgumentParser(description="Run Backtest Engine (skeleton)")
    parser.add_argument("--config-dir", default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--model-registry", default=None, help="Path to model registry; overrides config.models.registry_path")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after run (once wired)")

    args = parser.parse_args()

    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info(f"Loading configuration from {conf_dir} ...")
    loader = ConfigLoader(conf_dir)
    cfg = loader.load()

    registry_path = args.model_registry or cfg.get("models", {}).get("registry_path", "models/")
    registry = ModelRegistry(registry_path)

    engine = Backtester(loader, registry)
    LOG.info("Backtester initialized. Data pipeline not wired in CLI yet.")
    LOG.info("Provide asset_frames programmatically to Backtester.run().")

    if args.dashboard:
        LOG.info("Dashboard launch requested, but run() not executed. Wire data before enabling.")
        dash_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dash_path])


if __name__ == "__main__":
    main()
