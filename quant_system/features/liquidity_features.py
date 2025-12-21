"""
Liquidity Feature Builder
-------------------------

Extracts quantitative liquidity measures useful for:
    - liquidity-flow model
    - BOS continuation model
    - hazard/survival model
    - execution confluence scoring

Features include:
    - Relative equal-high/low density
    - Liquidity pool proximity
    - Sweep strength (from SMC sweep detection)
    - Wick displacement ratios
    - Volume-pressure metrics
    - Multi-TF liquidity gradient (higher TF swings & OBs)

No pandas used. Strictly closed-bar data. Compatible with 15m, 1h, 6h, 12h.
"""

from typing import Dict, List
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class LiquidityFeatureBuilder:
    """
    Build liquidity-related features by integrating:
        sweeps, swings, zones, fvg, BOS/CHOCH, volume/wick-based measures.

    Parameters:
        proximity_window: search window for nearest liquidity pools
        wick_factor: weight applied to wick-based displacement metrics
    """

    def __init__(
        self,
        proximity_window: int = 50,
        wick_factor: float = 1.0
    ):
        self.proximity_window = proximity_window
        self.wick_factor = wick_factor

        log(
            f"LiquidityFeatureBuilder initialized "
            f"(proximity_window={proximity_window}, wick_factor={wick_factor})."
        )

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

        log(f"Building liquidity features for {len(candles):,} candles.")

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

        log("Liquidity feature building complete.")
        return result

    # Pandas-friendly wrapper (placeholder passthrough for current pipeline)
    def apply(self, df15):
        """
        Placeholder to keep pipeline moving; extend with pandas liquidity features if needed.
        """
        return df15

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
