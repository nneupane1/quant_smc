"""Compatibility wrapper for canonical hazard labels."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import (
    compute_hazard_labels,
    frame_from_candles,
    map_series,
)
from quant_system.utils.logger import log


class HazardLabeler:
    def __init__(self, horizon_bars: int = 48, dd_r: float = 1.0):
        self.horizon_bars = horizon_bars
        self.dd_r = dd_r
        log(f"HazardLabeler initialized (horizon_bars={horizon_bars}, dd_r={dd_r}).")

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["hazard"])
        return {
            "horizon_bars": self.horizon_bars,
            "event_R_threshold": -abs(self.dd_r),
            "treat_choch_as_event": True,
            "treat_stop_as_event": True,
        }

    def generate_labels(
        self,
        candles: List[Candle],
        entries: Dict[int, Dict[str, float]],
        atr_15m: Dict[int, float],
        choch: Dict[int, Dict[str, float]],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        df = frame_from_candles(candles)
        df["atr_15m"] = map_series(df, atr_15m, "atr_15m")
        df["side"] = map_series(
            df,
            {k: ("long" if v.get("direction", 0) >= 0 else "short") for k, v in entries.items()},
            "side",
        )
        for col in ("choch_up", "choch_down"):
            df[col] = map_series(df, {k: v.get(col) for k, v in choch.items()}, col)
        events, times = compute_hazard_labels(df, self._cfg(None))
        ts = df["timestamp"].tolist()
        return dict(zip(ts, times.astype(int))), dict(zip(ts, events.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["hazard_event"], df["hazard_time"] = compute_hazard_labels(df, self._cfg(cfg_loader))
        return df
