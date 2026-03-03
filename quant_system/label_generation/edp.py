"""Compatibility wrapper for canonical EDP labels."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import compute_edp_labels, frame_from_candles, map_series
from quant_system.utils.logger import log


class EDPLabeler:
    def __init__(self, horizon_bars: int = 96, dd_r: float = 3.0):
        self.horizon_bars = horizon_bars
        self.dd_r = dd_r
        log(f"EDPLabeler initialized (horizon_bars={horizon_bars}, dd_r={dd_r}).")

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["edp"])
        return {"horizon_bars": self.horizon_bars, "drawdown_R_threshold": -abs(self.dd_r)}

    def generate_labels(
        self,
        candles: List[Candle],
        atr_15m: Dict[int, float],
    ) -> Dict[int, int]:
        df = frame_from_candles(candles)
        df["atr_15m"] = map_series(df, atr_15m, "atr_15m")
        labels = compute_edp_labels(df, self._cfg(None))
        return dict(zip(df["timestamp"], labels.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["label_edp"] = compute_edp_labels(df, self._cfg(cfg_loader))
        return df
