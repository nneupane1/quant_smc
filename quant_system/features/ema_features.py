"""
EMA Feature Builder
-------------------

Builds high-quality EMA-derived features for all TFs:
    - EMA values
    - Slopes (percent change between bars)
    - Distance-to-EMA normalized by ATR
    - Z-scored distance
    - EMA band regime (inside/outside ± k * ATR)
    - Alignment flags (multi-TF trend agreement)

Supports:
    15m, 1h, 6h, 12h
"""

from typing import Dict, List, Optional
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class EMAFeatureBuilder:
    """
    Compute EMA-based features for per-TF candle streams.

    Parameters:
        periods_by_tf: { "15m": [21,55], "1h":[50,200], ... }
        band_k_atr: width of EMA band = EMA ± k * ATR
        z_window: number of candles over which to compute z-score of distance
    """

    def __init__(
        self,
        periods_by_tf: Dict[str, List[int]],
        band_k_atr: float = 1.5,
        z_window: int = 200,
    ):
        self.periods_by_tf = periods_by_tf
        self.band_k_atr = band_k_atr
        self.z_window = z_window

        log(
            f"EMAFeatureBuilder initialized "
            f"(band_k_atr={band_k_atr}, z_window={z_window})."
        )

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

        log(f"Building EMA features for TF={tf}, count={len(candles):,}.")

        if tf not in self.periods_by_tf:
            log(f"No EMA periods specified for TF={tf}. Returning empty result.")
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

        log(f"EMA feature building complete for TF={tf}.")
        return result

    # Pandas-friendly wrapper (placeholder passthrough for current pipeline)
    def apply(self, df15, df1h=None, df6h=None, df12h=None):
        """
        Placeholder to keep pipeline moving; extend with pandas EMA features if needed.
        """
        return df15
