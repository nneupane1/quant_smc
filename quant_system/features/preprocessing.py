"""
Liquidity feature engineering:
    - Sweep flags (formal SMC sweeps)
    - Equal-high/low density
    - Liquidity cluster density
    - Wick pressure (buy/sell imbalance)
    - Rolling + lag logic via RollingWindows

Used by liquidity-flow model, BOS continuation, hazard engine,
and confluence scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from quant_system.utils.logger import log
from quant_system.features.rolling_windows import RollingWindows
from quant_system.config.config_loader import ConfigLoader


class LiquidityFeatures:
    """
    Multi-timeframe liquidity feature builder.
    Anchored to 15m rows but uses 1h/6h sweeps, equal highs/lows,
    wick-pressure and liquidity density.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["features"]["liquidity"]

        self.eql_window = int(self.cfg["eql_window"])
        self.eql_cluster_min = int(self.cfg["eql_cluster_min"])
        self.wick_pressure_lb = int(self.cfg["wick_pressure_lookback"])
        self.liq_density_win = int(self.cfg["liquidity_density_window"])

        log("LiquidityFeatures initialized.")

    # ------------------------------------------------------------
    # Sweep detection
    # ------------------------------------------------------------
    @staticmethod
    def _detect_sweep(df: pd.DataFrame) -> pd.Series:
        """
        Simple sweep flag:
            - High takes previous N highs OR
            - Low takes previous N lows
        More elaborate SMC sweeps happen upstream;
        here we use a binary flag for ML features.
        """
        highs = df["high"].shift(1)
        lows = df["low"].shift(1)

        sweep_up = (df["high"] > highs) & (df["high"] > df["high"].rolling(3).max().shift(1))
        sweep_dn = (df["low"] < lows) & (df["low"] < df["low"].rolling(3).min().shift(1))

        return (sweep_up | sweep_dn).astype(int)

    # ------------------------------------------------------------
    # Equal-high/low cluster density
    # ------------------------------------------------------------
    def _equal_level_density(self, df: pd.DataFrame) -> pd.Series:
        """
        Count number of equal-high / equal-low levels in the past window.
        """
        highs = df["high"].round(2)
        lows = df["low"].round(2)

        eql_high = highs.rolling(self.eql_window).apply(
            lambda x: pd.Series(x).value_counts().max(), raw=False
        )

        eql_low = lows.rolling(self.eql_window).apply(
            lambda x: pd.Series(x).value_counts().max(), raw=False
        )

        density = np.maximum(eql_high, eql_low)
        density = (density >= self.eql_cluster_min).astype(int)
        return density

    # ------------------------------------------------------------
    # Wick pressure (buy/sell imbalance)
    # ------------------------------------------------------------
    def _wick_pressure(self, df: pd.DataFrame) -> pd.Series:
        """
        wick_pressure = (upper_wick - lower_wick) normalized.
        Rolling z-scored to avoid scale bias.
        """
        upper = df["high"] - df["close"]
        lower = df["close"] - df["low"]
        raw = upper - lower

        z = raw.rolling(self.wick_pressure_lb).apply(
            lambda x: (x[-1] - x.mean()) / (x.std() if x.std() != 0 else np.nan),
            raw=False,
        )
        return z

    # ------------------------------------------------------------
    # Liquidity density (rolling range clustering)
    # ------------------------------------------------------------
    def _liquidity_density(self, df: pd.DataFrame) -> pd.Series:
        """
        Measures how often local highs/lows cluster around
        levels within a window → indicates liquidity pools.
        """
        levels = (df["high"].round(2) + df["low"].round(2)) / 2

        density = levels.rolling(self.liq_density_win).apply(
            lambda x: pd.Series(x).value_counts().max(), raw=False
        )

        z = (density - density.mean()) / density.std()
        return z

    # ------------------------------------------------------------
    # Combine all features into a single dataframe
    # ------------------------------------------------------------
    def build(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Build liquidity features for 15m frame:
            - sweep flag
            - equal-high/low density
            - wick pressure z-score
            - liquidity density z-score
            - lagged sweep flags (RollingWindows)
        """

        df = df_15m.copy()

        log("LiquidityFeatures: computing sweep flags...")
        df["liq_sweep"] = self._detect_sweep(df)

        log("LiquidityFeatures: computing EQL density...")
        df["liq_eql_density"] = self._equal_level_density(df)

        log("LiquidityFeatures: computing wick pressure...")
        df["liq_wick_pressure"] = self._wick_pressure(df)

        log("LiquidityFeatures: computing liquidity density...")
        df["liq_density"] = self._liquidity_density(df)

        # --------------------------------------------------------
        # LAGGED liquidity signals (for Liq-Flow model)
        # --------------------------------------------------------
        lag_cols = ["liq_sweep", "liq_eql_density", "liq_wick_pressure", "liq_density"]
        df = RollingWindows.add_lags(df, lag_cols, lags=[1, 2, 3])

        log("LiquidityFeatures: final feature set ready.")
        return df


class FeaturePreprocessor:
    """
    Minimal preprocessing wrapper for compatibility with FeatureBuilder.
    Optionally, this could handle scaling/winsorization; for now, it is a pass-through.
    """

    def __init__(self, config_loader: ConfigLoader = None):
        self.cfg_loader = config_loader
        self.cfg = {}
        if config_loader:
            try:
                self.cfg = config_loader.load_yaml("features.yaml").get("features", {}).get("scale", {})
            except Exception:
                self.cfg = {}
        log("FeaturePreprocessor initialized (pass-through).")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder for scaling/cleaning; currently returns df unchanged.
        return df
