"""Compatibility wrapper for canonical micro-momentum labels."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import compute_momo_labels, frame_from_candles, map_series
from quant_system.utils.logger import log


class MicroMomentumLabeler:
    def __init__(self, horizon_bars: int = 6, noise_k: float = 0.3):
        self.horizon_bars = horizon_bars
        self.noise_k = noise_k
        log(
            f"MicroMomentumLabeler initialized "
            f"(horizon_bars={horizon_bars}, noise_k={noise_k})."
        )

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["momo"])
        return {
            "min_horizon": max(self.horizon_bars - 2, 1),
            "max_horizon": self.horizon_bars,
            "noise_band_sigma": self.noise_k,
            "return_threshold_sigma": 1.2,
        }

    def generate_labels(
        self,
        candles: List[Candle],
        atr_15m: Dict[int, float],
    ) -> Dict[int, int]:
        df = frame_from_candles(candles)
        df["atr_15m"] = map_series(df, atr_15m, "atr_15m")
        labels = compute_momo_labels(df, self._cfg(None))
        return dict(zip(df["timestamp"], labels.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["label_momo"] = compute_momo_labels(df, self._cfg(cfg_loader))
        return df
