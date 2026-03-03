"""Compatibility loader that delegates to the canonical label builder."""

from __future__ import annotations

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.label_builder import LabelBuilder
from quant_system.utils.logger import get_logger

LOG = get_logger("label_loader")


class LabelLoader:
    """Legacy training API retained for older imports."""

    def __init__(self, config_loader: ConfigLoader):
        self.builder = LabelBuilder(config_loader)
        LOG.info("[LabelLoader] Initialized via canonical LabelBuilder.")

    def build(self, df15: pd.DataFrame, asset: str) -> pd.DataFrame:
        LOG.info("[LabelLoader] Building labels for asset=%s", asset)
        df = self.builder.apply(df15.copy()).reset_index(drop=True)
        LOG.info("[LabelLoader] Completed labels rows=%s asset=%s", len(df), asset)
        return df
