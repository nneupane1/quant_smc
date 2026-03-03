"""Canonical dataframe-first label builder for the 15m execution spine."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.utils import (
    compute_bos_cont_labels,
    compute_edp_labels,
    compute_eop_labels,
    compute_flow_1h_labels,
    compute_hazard_labels,
    compute_liq_flow_labels,
    compute_momo_labels,
)
from quant_system.utils.logger import get_logger

LOG = get_logger("label_builder")


class LabelBuilder:
    def __init__(self, config_loader: ConfigLoader):
        self.cfg_loader = config_loader
        self.labels_cfg = config_loader.load_yaml("labels.yaml")["labels"]

    def apply(self, df15: pd.DataFrame) -> pd.DataFrame:
        df = df15.copy()
        df["label_liq_flow"] = compute_liq_flow_labels(df, self.labels_cfg["liq_flow"])
        df["label_bos_cont"] = compute_bos_cont_labels(df, self.labels_cfg["bos_cont"])
        df["label_momo"] = compute_momo_labels(df, self.labels_cfg["momo"])
        df["label_flow_1h"] = compute_flow_1h_labels(df, self.labels_cfg.get("flow_1h", {}))
        df["label_eop"] = compute_eop_labels(df, self.labels_cfg["eop"])
        df["label_edp"] = compute_edp_labels(df, self.labels_cfg["edp"])
        df["hazard_event"], df["hazard_time"] = compute_hazard_labels(df, self.labels_cfg["hazard"])
        return df


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate labels from a 15m feature CSV.")
    parser.add_argument("--config", default="quant_system/config", help="Config directory")
    parser.add_argument("--features", required=True, help="Input features CSV")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    lb = LabelBuilder(cfg)

    LOG.info("[LabelBuilder] Loading features from %s", args.features)
    df = pd.read_csv(args.features)
    LOG.info("[LabelBuilder] Generating labels")
    labels = lb.apply(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    LOG.info("[LabelBuilder] Wrote labels to %s", out_path)


if __name__ == "__main__":
    _cli()
