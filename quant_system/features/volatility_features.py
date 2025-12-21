"""
Volatility feature engineering:
    - ATR(14)
    - Realized volatility
    - ATR slope
    - Range% compression
    - Volatility z-scores
    - Rolling/lag logic via RollingWindows

Used by BOS continuation, liquidity-flow, momentum, hazard engine,
and regime inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from quant_system.utils.logger import log
from quant_system.features.rolling_windows import RollingWindows
from quant_system.config.config_loader import ConfigLoader


class VolatilityFeatures:
    """
    Build volatility features for any timeframe.
    Anchored primarily to 15m but equally valid on 1h/6h/12h frames.
    """

    def __init__(self, config: Dict[str, Any]):
        vcfg = config["features"]["volatility"]

        self.atr_period = int(vcfg["atr_period"])
        self.realized_vol_lb = int(vcfg["realized_vol_lookback"])
        self.atr_slope_lb = int(vcfg["atr_slope_lookback"])
        self.range_pct_lb = int(vcfg["range_pct_lookback"])
        self.vol_z_lb = int(vcfg["volatility_z_window"])

        log("VolatilityFeatures initialized.")

    # ------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------
    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)

        tr = pd.DataFrame({
            "h-l": high - low,
            "h-c": (high - prev_close).abs(),
            "l-c": (low - prev_close).abs(),
        }).max(axis=1)

        atr = tr.rolling(self.atr_period).mean()
        return atr

    # ------------------------------------------------------------
    # Realized volatility
    # ------------------------------------------------------------
    def _realized_vol(self, df: pd.DataFrame) -> pd.Series:
        ret = df["close"].pct_change()
        vol = ret.rolling(self.realized_vol_lb).std()
        return vol

    # ------------------------------------------------------------
    # ATR slope
    # ------------------------------------------------------------
    def _atr_slope(self, atr: pd.Series) -> pd.Series:
        slope = (atr - atr.shift(1)) / atr.shift(1)
        return slope.rolling(self.atr_slope_lb).mean()

    # ------------------------------------------------------------
    # Range% (compression indicator)
    # ------------------------------------------------------------
    def _range_pct(self, df: pd.DataFrame) -> pd.Series:
        rng = df["high"] - df["low"]
        close = df["close"]
        pct = (rng / close).rolling(self.range_pct_lb).mean()
        return pct

    # ------------------------------------------------------------
    # Volatility z-score
    # ------------------------------------------------------------
    def _vol_z(self, vol: pd.Series) -> pd.Series:
        mean = vol.rolling(self.vol_z_lb).mean()
        std = vol.rolling(self.vol_z_lb).std()
        return (vol - mean) / std.replace(0, np.nan)

    # ------------------------------------------------------------
    # Combine features
    # ------------------------------------------------------------
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build volatility features:
            atr
            realized_vol
            atr_slope
            range_pct
            vol_zscore
            lag features
        """

        log("VolatilityFeatures: computing ATR...")
        atr = self._atr(df)

        log("VolatilityFeatures: computing realized vol...")
        rvol = self._realized_vol(df)

        log("VolatilityFeatures: computing ATR slope...")
        atr_slope = self._atr_slope(atr)

        log("VolatilityFeatures: computing range %...")
        r_pct = self._range_pct(df)

        log("VolatilityFeatures: computing volatility z-score...")
        v_z = self._vol_z(rvol)

        out = df.copy()
        out["atr"] = atr
        out["realized_vol"] = rvol
        out["atr_slope"] = atr_slope
        out["range_pct"] = r_pct
        out["vol_zscore"] = v_z

        # RollingWindow lag logic
        lag_cols = ["atr", "realized_vol", "atr_slope", "range_pct", "vol_zscore"]
        out = RollingWindows.add_lags(out, lag_cols, lags=[1, 2, 3])

        log("VolatilityFeatures: final feature set ready.")
        return out


class VolatilityFeatureBuilder:
    """
    Thin wrapper to align with the FeatureBuilder API.
    Loads volatility config via ConfigLoader if one is not provided.
    """

    def __init__(self, config_loader: ConfigLoader = None):
        if config_loader is None:
            conf_dir = str((Path(__file__).resolve().parents[1] / "config"))
            config_loader = ConfigLoader(conf_dir)
        self.cfg = config_loader.load_yaml("features.yaml")
        self.vol_cfg = self.cfg.get("features", {}).get("volatility", {})
        self.block = VolatilityFeatures({"features": {"volatility": self.vol_cfg}})

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volatility features to the provided dataframe.
        """
        return self.block.build(df)
