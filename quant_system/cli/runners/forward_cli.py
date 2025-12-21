"""
CLI wrapper for forward-testing (paper mode).

Note: The streaming pipeline is not wired in this CLI yet; use programmatic integration.
"""

import argparse
import os
import sys
import subprocess

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("forward_cli")


def resolve_conf_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def main():
    parser = argparse.ArgumentParser(description="Run Forward Test (skeleton)")
    parser.add_argument("--config-dir", default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after run (once wired)")

    args = parser.parse_args()

    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info(f"Loading configuration from {conf_dir} ...")
    loader = ConfigLoader(conf_dir)
    loader.load()  # validate

    LOG.info("ForwardEngine initialization is deferred until data/stream wiring is complete.")

    if args.dashboard:
        LOG.info("Dashboard launch requested, but run() not executed. Wire data before enabling.")
        dash_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "forward_monitor.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dash_path])


if __name__ == "__main__":
    main()
