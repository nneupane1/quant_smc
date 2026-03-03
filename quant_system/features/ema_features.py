"""EMA feature engineering with deterministic pandas and candle-list paths."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import get_logger

LOG = get_logger("ema_features")


class EMAFeatureBuilder:
    """
    Compute EMA-based features for per-TF candle streams.

    Accepted constructor inputs:
      - `ConfigLoader`
      - a merged config dict with `features.ema`
      - a direct `{tf: [periods]}` mapping
    """

    def __init__(
        self,
        periods_by_tf: Optional[Any] = None,
        band_k_atr: float = 1.5,
        z_window: int = 200,
    ):
        cfg = self._resolve_cfg(periods_by_tf)
        ema_cfg = cfg.get("features", {}).get("ema", {}) if isinstance(cfg, dict) else {}

        direct_periods = periods_by_tf if isinstance(periods_by_tf, dict) and "features" not in periods_by_tf else None
        self.periods_by_tf = direct_periods or ema_cfg.get(
            "periods",
            {"15m": [21, 55], "1h": [50, 200], "6h": [100], "12h": [200]},
        )
        self.band_k_atr = float(ema_cfg.get("band_k_atr", band_k_atr))
        self.z_window = int(ema_cfg.get("z_window", z_window))

        LOG.info(
            "EMAFeatureBuilder initialized (band_k_atr=%s, z_window=%s).",
            self.band_k_atr,
            self.z_window,
        )

    @staticmethod
    def _resolve_cfg(config_like: Optional[Any]) -> Dict[str, Any]:
        if isinstance(config_like, ConfigLoader):
            return config_like.load_yaml("features.yaml")
        if isinstance(config_like, dict):
            return config_like
        return {}

    def _ema(self, prev: float, price: float, period: int) -> float:
        k = 2.0 / (period + 1.0)
        return price * k + prev * (1 - k)

    def _atr(self, candles: List[Candle], i: int, period: int = 14) -> float:
        if i == 0:
            return candles[0].high - candles[0].low
        trs = []
        for j in range(max(0, i - period + 1), i + 1):
            c = candles[j]
            prev_close = candles[j - 1].close if j > 0 else c.close
            tr = max(
                c.high - c.low,
                abs(c.high - prev_close),
                abs(c.low - prev_close),
            )
            trs.append(tr)
        return sum(trs) / len(trs)

    def build(self, tf: str, candles: List[Candle]) -> Dict[int, Dict[str, float]]:
        """
        Compute EMA features for a single timeframe.

        Returns:
            ts → {
                "ema_{p}": float,
                "ema_slope_{p}": float,
                "dist_ema_{p}": float,
                "z_dist_ema_{p}": float,
                "band_regime_{p}": float,
            }
        """

        LOG.info("Building EMA features for TF=%s, count=%s.", tf, len(candles))

        if tf not in self.periods_by_tf:
            LOG.info("No EMA periods specified for TF=%s. Returning empty result.", tf)
            return {}

        periods = self.periods_by_tf[tf]
        ts_arr = [c.timestamp for c in candles]

        # Pre-allocate EMA trackers
        ema_vals = {p: [] for p in periods}
        atr_vals = []
        z_buffers = {p: [] for p in periods}

        # Compute ATR first to normalize distance-to-EMA
        for i in range(len(candles)):
            atr_vals.append(self._atr(candles, i))

        # Compute EMAs
        for p in periods:
            prev = candles[0].close
            ema_vals[p].append(prev)
            for i in range(1, len(candles)):
                prev = self._ema(prev, candles[i].close, p)
                ema_vals[p].append(prev)

        result: Dict[int, Dict[str, float]] = {}

        # Feature assembly
        for i, c in enumerate(candles):
            ts = ts_arr[i]
            rec = {}

            for p in periods:
                ema_v = ema_vals[p][i]
                atr = atr_vals[i]

                slope = 0.0
                if i > 0:
                    prev_ema = ema_vals[p][i - 1]
                    slope = (ema_v - prev_ema) / max(1e-9, prev_ema)

                dist = (c.close - ema_v) / max(1e-9, atr)

                zb = z_buffers[p]
                zb.append(dist)
                if len(zb) > self.z_window:
                    zb.pop(0)

                mean_z = sum(zb) / len(zb)
                var = sum((d - mean_z) ** 2 for d in zb) / max(1, len(zb) - 1)
                std_z = var ** 0.5 if var > 0 else 1e-9
                zdist = (dist - mean_z) / std_z

                band_upper = ema_v + self.band_k_atr * atr
                band_lower = ema_v - self.band_k_atr * atr

                if c.close > band_upper:
                    regime = 1.0
                elif c.close < band_lower:
                    regime = -1.0
                else:
                    regime = 0.0

                rec[f"ema_{p}"] = ema_v
                rec[f"ema_slope_{p}"] = slope
                rec[f"dist_ema_{p}"] = dist
                rec[f"z_dist_ema_{p}"] = zdist
                rec[f"band_regime_{p}"] = regime

            result[ts] = rec

        LOG.info("EMA feature building complete for TF=%s.", tf)
        return result

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [c.lower() for c in frame.columns]
        if "dt" not in frame.columns:
            if "timestamp" in frame.columns:
                frame["dt"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
            else:
                raise ValueError("EMAFeatureBuilder.apply requires `dt` or `timestamp`.")
        frame["dt"] = pd.to_datetime(frame["dt"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.sort_values("dt").reset_index(drop=True)

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_close = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - prev_close).abs(),
                (frame["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def _build_tf_frame(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        frame = self._normalize_frame(df)
        periods = [int(p) for p in self.periods_by_tf.get(tf, [])]
        if not periods:
            return frame[["dt"]].copy()

        out = frame[["dt"]].copy()
        atr = frame["atr"] if "atr" in frame.columns else self._atr(frame)
        close = frame["close"]

        for p in periods:
            ema = close.ewm(span=p, adjust=False, min_periods=1).mean()
            slope = ema.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            dist = ((close - ema) / atr.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            z = (
                (dist - dist.rolling(self.z_window, min_periods=10).mean())
                / dist.rolling(self.z_window, min_periods=10).std().replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

            upper = ema + self.band_k_atr * atr
            lower = ema - self.band_k_atr * atr
            regime = pd.Series(0.0, index=frame.index)
            regime = regime.mask(close > upper, 1.0)
            regime = regime.mask(close < lower, -1.0)

            out[f"ema_{p}_{tf}"] = ema
            out[f"ema_slope_{p}_{tf}"] = slope
            out[f"dist_ema_{p}_{tf}"] = dist
            out[f"z_dist_ema_{p}_{tf}"] = z
            out[f"band_regime_{p}_{tf}"] = regime

        fast = periods[0]
        slow = periods[-1]
        out[f"ema_fast_{tf}"] = out[f"ema_{fast}_{tf}"]
        out[f"ema_slow_{tf}"] = out[f"ema_{slow}_{tf}"]
        out[f"dist_to_ema_{tf}"] = out[f"dist_ema_{fast}_{tf}"]
        out[f"band_regime_{tf}"] = out[f"band_regime_{fast}_{tf}"]
        out[f"ema_alignment_{tf}"] = (
            (out[f"ema_fast_{tf}"] > out[f"ema_slow_{tf}"]).astype(int)
            - (out[f"ema_fast_{tf}"] < out[f"ema_slow_{tf}"]).astype(int)
        )
        return out

    @staticmethod
    def _join(anchor: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
        if ctx is None or ctx.empty:
            return anchor
        return pd.merge_asof(
            anchor.sort_values("dt"),
            ctx.sort_values("dt"),
            on="dt",
            direction="backward",
            allow_exact_matches=False,
        )

    def apply(
        self,
        df15: pd.DataFrame,
        df1h: Optional[pd.DataFrame] = None,
        df6h: Optional[pd.DataFrame] = None,
        df12h: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build EMA-derived features on each timeframe and project higher-TF EMA
        state onto the 15m execution spine.
        """
        if df15 is None or df15.empty:
            return df15

        out = self._build_tf_frame(df15, "15m")
        base = self._normalize_frame(df15)
        merged = base.merge(out, on="dt", how="left")

        for tf, frame in (("1h", df1h), ("6h", df6h), ("12h", df12h)):
            if frame is None or frame.empty:
                continue
            merged = self._join(merged, self._build_tf_frame(frame, tf))

        periods_15m = [int(p) for p in self.periods_by_tf.get("15m", [])]
        if periods_15m:
            fast = periods_15m[0]
            slow = periods_15m[-1]
            merged["ema_fast"] = merged[f"ema_{fast}_15m"]
            merged["ema_slow"] = merged[f"ema_{slow}_15m"]
            merged["dist_ema"] = merged[f"dist_ema_{fast}_15m"]
            merged["dist_to_ema"] = merged[f"dist_ema_{fast}_15m"]
            merged["band_regime"] = merged[f"band_regime_{fast}_15m"]

        if "ema_fast_1h" in merged.columns and "ema_slow_1h" in merged.columns:
            merged["ema_rel_1h"] = (
                (merged["ema_fast_1h"] - merged["ema_slow_1h"])
                / merged["ema_slow_1h"].replace(0, np.nan)
            )

        if "timestamp" not in merged.columns:
            merged["timestamp"] = (pd.to_datetime(merged["dt"], utc=True).astype("int64") // 10**9).astype("int64")
        return merged
