"""
CLI wrapper for live trading orchestration.

Note: Live orchestration is not wired in this CLI yet; configure connectors and streams before enabling.
"""

import argparse
import os
import sys
import subprocess

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("live_cli")


def resolve_conf_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def main():
    parser = argparse.ArgumentParser(description="Run Live Trading (skeleton)")
    parser.add_argument("--config-dir", default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after run (once wired)")

    args = parser.parse_args()

    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info(f"Loading configuration from {conf_dir} ...")
    loader = ConfigLoader(conf_dir)
    loader.load()  # validate

    LOG.info("LiveOrchestrator initialization is deferred until exchange connectors/streams are wired.")

    if args.dashboard:
        LOG.info("Dashboard launch requested, but run() not executed. Wire live pipelines before enabling.")
        dash_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "live_monitor.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dash_path])


if __name__ == "__main__":
    main()
