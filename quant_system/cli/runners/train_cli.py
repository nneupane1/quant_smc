"""
CLI wrapper for training orchestration.

Note: The full training pipeline needs data/feature/label wiring before activation.
"""

import argparse
import os

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("train_cli")


def resolve_conf_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def main():
    parser = argparse.ArgumentParser(description="Train all models (skeleton)")
    parser.add_argument("--config-dir", default=os.path.join(os.path.dirname(__file__), "..", "config"))

    args = parser.parse_args()

    conf_dir = resolve_conf_dir(args.config_dir)
    LOG.info(f"Loading configuration from {conf_dir} ...")
    loader = ConfigLoader(conf_dir)
    loader.load()  # validate

    LOG.info("Training pipeline not wired in CLI yet. Implement data/feature/label/training orchestration before use.")


if __name__ == "__main__":
    main()
