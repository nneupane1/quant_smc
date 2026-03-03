"""Compatibility wrapper for canonical BOS continuation labels."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import compute_bos_cont_labels, frame_from_candles, map_series
from quant_system.utils.logger import log


class BOSContinuationLabeler:
    def __init__(self, horizon_bars: int = 48, reward_r: float = 3.0):
        self.horizon_bars = horizon_bars
        self.reward_r = reward_r
        log(
            f"BOSContinuationLabeler initialized "
            f"(horizon_bars={horizon_bars}, reward_r={reward_r})."
        )

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["bos_cont"])
        return {
            "horizon_bars": self.horizon_bars,
            "min_R": self.reward_r,
            "bos_atr_buffer_mult": 0.0,
            "invalidation_on_choch": True,
        }

    def generate_labels(
        self,
        candles: List[Candle],
        smc: Dict[int, Dict[str, float]],
        atr: Dict[int, float],
    ) -> Dict[int, int]:
        df = frame_from_candles(candles)
        for col in ("bos_up", "bos_down", "choch_up", "choch_down", "broken_level"):
            df[col] = map_series(df, {k: v.get(col) for k, v in smc.items()}, col)
        df["atr_15m"] = map_series(df, atr, "atr_15m")
        labels = compute_bos_cont_labels(df, self._cfg(None))
        return dict(zip(df["timestamp"], labels.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["label_bos_cont"] = compute_bos_cont_labels(df, self._cfg(cfg_loader))
        return df
