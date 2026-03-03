"""Compatibility wrapper for canonical EOP labels."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import compute_eop_labels, frame_from_candles, map_series
from quant_system.utils.logger import log


class EOPLabeler:
    def __init__(
        self,
        horizon_bars: int = 96,
        tau_A_plus: float = 0.80,
        evr_min: float = 1.5,
        median_r_min: float = 5.0,
        hazard_max: float = 0.35,
    ):
        self.horizon_bars = horizon_bars
        self.tau_A_plus = tau_A_plus
        self.evr_min = evr_min
        self.median_r_min = median_r_min
        self.hazard_max = hazard_max
        log(
            "EOPLabeler initialized "
            f"(horizon_bars={horizon_bars}, tau_A_plus={tau_A_plus}, "
            f"evr_min={evr_min}, median_r_min={median_r_min}, hazard_max={hazard_max})."
        )

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["eop"])
        return {
            "horizon_bars": self.horizon_bars,
            "Aplus_min_conf": self.tau_A_plus,
            "Aplus_min_evr": self.evr_min,
            "Aplus_min_medianR": self.median_r_min,
            "hazard_cap": self.hazard_max,
        }

    def generate_labels(
        self,
        candles: List[Candle],
        conf_scores: Dict[int, float],
        evr_scores: Dict[int, float],
        median_r: Dict[int, float],
        hazard: Dict[int, float],
        tf_gates: Dict[int, Dict[str, float]],
    ) -> Dict[int, int]:
        df = frame_from_candles(candles)
        df["conf_score"] = map_series(df, conf_scores, "conf_score")
        df["evr"] = map_series(df, evr_scores, "evr")
        df["median_r"] = map_series(df, median_r, "median_r")
        df["hazard"] = map_series(df, hazard, "hazard")
        allow_map = {k: ("A+" if (v.get("allow_long") or v.get("allow_short")) else "") for k, v in tf_gates.items()}
        df["tier"] = map_series(df, allow_map, "tier")
        labels = compute_eop_labels(df, self._cfg(None))
        return dict(zip(df["timestamp"], labels.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["label_eop"] = compute_eop_labels(df, self._cfg(cfg_loader))
        return df
