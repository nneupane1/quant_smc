"""Liquidity feature engineering with pandas and legacy candle-list compatibility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.features.rolling_windows import RollingWindows
from quant_system.utils.logger import get_logger

LOG = get_logger("liquidity_features")


class LiquidityFeatureBuilder:
    """
    Build liquidity-related features by integrating:
        sweeps, swings, zones, fvg, BOS/CHOCH, volume/wick-based measures.

    Parameters:
        proximity_window: search window for nearest liquidity pools
        wick_factor: weight applied to wick-based displacement metrics
    """

    def __init__(self, config_like: Optional[Any] = None, proximity_window: int = 50, wick_factor: float = 1.0):
        cfg = self._resolve_cfg(config_like)
        liq_cfg = cfg.get("features", {}).get("liquidity", {}) if isinstance(cfg, dict) else {}

        self.proximity_window = int(liq_cfg.get("eql_window", proximity_window))
        self.eql_window = int(liq_cfg.get("eql_window", 50))
        self.eql_cluster_min = int(liq_cfg.get("eql_cluster_min", 2))
        self.wick_pressure_lb = int(liq_cfg.get("wick_pressure_lookback", 14))
        self.liq_density_win = int(liq_cfg.get("liquidity_density_window", 60))
        self.wick_factor = float(wick_factor)

        LOG.info(
            "LiquidityFeatureBuilder initialized (proximity_window=%s, wick_factor=%s).",
            self.proximity_window,
            self.wick_factor,
        )

    @staticmethod
    def _resolve_cfg(config_like: Optional[Any]) -> Dict[str, Any]:
        if isinstance(config_like, ConfigLoader):
            return config_like.load_yaml("features.yaml")
        if isinstance(config_like, dict):
            return config_like
        return {}

    def build(
        self,
        candles: List[Candle],
        swings: Dict[int, Dict[str, float]],
        sweeps: Dict[int, Dict[str, float]],
        zones: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Build liquidity features for each timestamp.

        Returns:
            ts → {
                "liq_eq_high_density": float,
                "liq_eq_low_density": float,
                "liq_near_pool_dist": float,
                "liq_sweep_strength": float,
                "liq_displacement_ratio": float,
                "liq_volume_pressure": float,
            }
        """

        LOG.info("Building liquidity features for %s candles.", len(candles))

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        ts_arr = [c.timestamp for c in candles]

        result: Dict[int, Dict[str, float]] = {}

        for i, c in enumerate(candles):
            ts = ts_arr[i]

            eq_high_density = 0.0
            eq_low_density = 0.0
            sweep_strength = 0.0
            displacement_ratio = 0.0
            volume_pressure = 0.0
            nearest_pool_distance = 0.0

            sw = swings.get(ts, {})
            sp = sweeps.get(ts, {})
            zn = zones.get(ts, {})

            # 1. Equal-high / equal-low density (liquidity pools)
            if sw.get("equal_high"):
                eq_high_density += 1.0
            if sw.get("equal_low"):
                eq_low_density += 1.0

            # 2. Sweep strength
            if sp.get("sweep_strength") is not None:
                sweep_strength = sp["sweep_strength"]

            # 3. Wick displacement ratio
            upper_wick = c.high - max(c.open, c.close)
            lower_wick = min(c.open, c.close) - c.low
            total_range = max(1e-9, c.high - c.low)
            displacement_ratio = self.wick_factor * abs(upper_wick - lower_wick) / total_range

            # 4. Volume pressure
            prev_vol = volumes[i - 1] if i > 0 else volumes[i]
            volume_pressure = (volumes[i] - prev_vol) / max(1e-9, prev_vol)

            # 5. Distance to nearest liquidity pool
            nearest_pool_distance = self._nearest_pool_distance(
                i, highs, lows, swings, ts_arr, self.proximity_window
            )

            result[ts] = {
                "liq_eq_high_density": eq_high_density,
                "liq_eq_low_density": eq_low_density,
                "liq_near_pool_dist": nearest_pool_distance,
                "liq_sweep_strength": sweep_strength,
                "liq_displacement_ratio": displacement_ratio,
                "liq_volume_pressure": volume_pressure,
            }

        LOG.info("Liquidity feature building complete.")
        return result

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [c.lower() for c in frame.columns]
        if "dt" not in frame.columns and "timestamp" in frame.columns:
            frame["dt"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        if "dt" in frame.columns:
            frame["dt"] = pd.to_datetime(frame["dt"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                frame[col] = np.nan
        return frame.sort_values("dt").reset_index(drop=True)

    def _equal_density(self, series: pd.Series) -> pd.Series:
        rounded = series.round(2)
        density = rounded.rolling(self.eql_window, min_periods=2).apply(
            lambda x: float(pd.Series(x).value_counts().max()),
            raw=False,
        )
        return density.fillna(0.0)

    def _wick_pressure(self, df: pd.DataFrame) -> pd.Series:
        upper = df["high"] - df[["open", "close"]].max(axis=1)
        lower = df[["open", "close"]].min(axis=1) - df["low"]
        raw = (upper - lower) * self.wick_factor
        mean = raw.rolling(self.wick_pressure_lb, min_periods=3).mean()
        std = raw.rolling(self.wick_pressure_lb, min_periods=3).std().replace(0, np.nan)
        return ((raw - mean) / std).replace([np.inf, -np.inf], np.nan)

    def _liquidity_density(self, df: pd.DataFrame) -> pd.Series:
        levels = ((df["high"] + df["low"]) / 2.0).round(2)
        density = levels.rolling(self.liq_density_win, min_periods=4).apply(
            lambda x: float(pd.Series(x).value_counts().max()),
            raw=False,
        )
        mean = density.rolling(self.liq_density_win, min_periods=4).mean()
        std = density.rolling(self.liq_density_win, min_periods=4).std().replace(0, np.nan)
        return ((density - mean) / std).replace([np.inf, -np.inf], np.nan)

    def _nearest_pool_distance_series(self, df: pd.DataFrame) -> pd.Series:
        prior_high = (
            pd.to_numeric(df["swing_high"], errors="coerce").ffill()
            if "swing_high" in df.columns
            else df["high"].rolling(self.eql_window).max().shift(1)
        )
        prior_low = (
            pd.to_numeric(df["swing_low"], errors="coerce").ffill()
            if "swing_low" in df.columns
            else df["low"].rolling(self.eql_window).min().shift(1)
        )
        pool_dist = pd.concat(
            [
                (df["close"] - prior_high).abs(),
                (df["close"] - prior_low).abs(),
            ],
            axis=1,
        ).min(axis=1)

        zone_dists = []
        for hi_col, lo_col in (
            ("demand_top", "demand_bottom"),
            ("supply_top", "supply_bottom"),
            ("zone_hi", "zone_lo"),
        ):
            if hi_col in df.columns and lo_col in df.columns:
                mid = (pd.to_numeric(df[hi_col], errors="coerce") + pd.to_numeric(df[lo_col], errors="coerce")) / 2.0
                zone_dists.append((df["close"] - mid).abs())

        if zone_dists:
            zone_dist = pd.concat(zone_dists, axis=1).min(axis=1)
            pool_dist = pd.concat([pool_dist, zone_dist], axis=1).min(axis=1)
        return pool_dist.replace([np.inf, -np.inf], np.nan)

    def apply(self, df15: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical pandas feature path used by training, backtest, forward, and dashboard
        reasoning. Produces liquidity diagnostics on the 15m execution spine.
        """
        if df15 is None or df15.empty:
            return df15

        df = self._normalize_frame(df15)
        out = df.copy()

        out["liq_eq_high_density"] = self._equal_density(out["high"])
        out["liq_eq_low_density"] = self._equal_density(out["low"])
        out["liq_eql_density"] = np.maximum(out["liq_eq_high_density"], out["liq_eq_low_density"])

        sweep_flag = pd.Series(0, index=out.index, dtype=int)
        if "sweep_flag" in out.columns:
            sweep_flag = pd.to_numeric(out["sweep_flag"], errors="coerce").fillna(0).astype(int)
        else:
            high_take = out["high"] > out["high"].rolling(3, min_periods=2).max().shift(1)
            low_take = out["low"] < out["low"].rolling(3, min_periods=2).min().shift(1)
            sweep_flag = (high_take | low_take).fillna(False).astype(int)
        out["liq_sweep"] = sweep_flag

        if "sweep_strength" in out.columns:
            out["liq_sweep_strength"] = pd.to_numeric(out["sweep_strength"], errors="coerce").fillna(0.0)
        else:
            prior_high = out["high"].rolling(3, min_periods=2).max().shift(1)
            prior_low = out["low"].rolling(3, min_periods=2).min().shift(1)
            sweep_up = (out["high"] - prior_high).clip(lower=0.0)
            sweep_dn = (prior_low - out["low"]).clip(lower=0.0)
            out["liq_sweep_strength"] = np.maximum(sweep_up.fillna(0.0), sweep_dn.fillna(0.0))
        out["liq_sweep_intensity"] = out["liq_sweep_strength"]

        upper_wick = out["high"] - out[["open", "close"]].max(axis=1)
        lower_wick = out[["open", "close"]].min(axis=1) - out["low"]
        total_range = (out["high"] - out["low"]).replace(0, np.nan)
        out["liq_displacement_ratio"] = (
            self.wick_factor * (upper_wick - lower_wick).abs() / total_range
        ).replace([np.inf, -np.inf], np.nan)
        out["liq_wick_pressure"] = self._wick_pressure(out)

        prev_vol = out["volume"].shift(1).replace(0, np.nan)
        out["liq_volume_pressure"] = ((out["volume"] - prev_vol) / prev_vol).replace([np.inf, -np.inf], np.nan)
        out["liq_density"] = self._liquidity_density(out)
        out["liq_near_pool_dist"] = self._nearest_pool_distance_series(out)

        lag_cols = [
            "liq_sweep",
            "liq_eql_density",
            "liq_sweep_strength",
            "liq_wick_pressure",
            "liq_density",
            "liq_near_pool_dist",
        ]
        out = RollingWindows.add_lags(out, lag_cols, lags=[1, 2, 3])
        return out

    def _nearest_pool_distance(
        self,
        idx: int,
        highs: List[float],
        lows: List[float],
        swings: Dict[int, Dict[str, float]],
        ts_arr: List[int],
        window: int
    ) -> float:
        """
        Find the nearest equal high/low liquidity pool within a window of bars.
        Returns distance in price terms.
        """

        lo = max(0, idx - window)
        hi = min(len(highs) - 1, idx + window)

        cur_price = (highs[idx] + lows[idx]) / 2.0
        best_dist = 1e9

        for j in range(lo, hi + 1):
            sw = swings.get(ts_arr[j], {})
            if sw.get("equal_high"):
                d = abs(highs[j] - cur_price)
                best_dist = min(best_dist, d)
            if sw.get("equal_low"):
                d = abs(lows[j] - cur_price)
                best_dist = min(best_dist, d)

        return best_dist if best_dist < 1e9 else 0.0
