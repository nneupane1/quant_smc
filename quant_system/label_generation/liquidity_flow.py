"""Compatibility wrapper for canonical liquidity-flow labels."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.label_generation.utils import compute_liq_flow_labels, frame_from_candles, map_series
from quant_system.utils.logger import log


class LiquidityFlowLabeler:
    def __init__(self, horizon_bars: int = 12, reward_r: float = 1.0):
        self.horizon_bars = horizon_bars
        self.reward_r = reward_r
        log(
            f"LiquidityFlowLabeler initialized "
            f"(horizon_bars={horizon_bars}, reward_r={reward_r})."
        )

    def _cfg(self, cfg_loader: Optional[ConfigLoader]) -> Dict[str, float]:
        if cfg_loader:
            return dict(cfg_loader.load_yaml("labels.yaml")["labels"]["liq_flow"])
        return {
            "horizon_bars": self.horizon_bars,
            "continuation_min_R": self.reward_r,
            "smc_requirements": {"require_sweep": True, "require_displacement": True},
        }

    def generate_labels(
        self,
        candles: List[Candle],
        sweeps: Dict[int, Dict[str, float]],
        displacement: Dict[int, Dict[str, float]],
        atr_15m: Dict[int, float],
    ) -> Dict[int, int]:
        df = frame_from_candles(candles)
        for col in ("sweep_high", "sweep_low", "swept_level", "sweep_strength", "displacement_hint"):
            df[col] = map_series(df, {k: v.get(col) for k, v in sweeps.items()}, col)
        for col in ("body_ratio", "vol_z", "displacement"):
            alias = {"body_ratio": "displacement_body_pct_15m", "vol_z": "volume_z_15m", "displacement": "displacement_flag_15m"}[col]
            df[alias] = map_series(df, {k: v.get(col) for k, v in displacement.items()}, alias)
        df["atr_15m"] = map_series(df, atr_15m, "atr_15m")
        labels = compute_liq_flow_labels(df, self._cfg(None))
        return dict(zip(df["timestamp"], labels.astype(int)))

    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        df = df15.copy()
        df["label_liq_flow"] = compute_liq_flow_labels(df, self._cfg(cfg_loader))
        return df
