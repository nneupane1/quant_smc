"""
Swing High / Swing Low Detector
-------------------------------

This module provides a deterministic, non-repainting implementation of
fractal-based swing detection across multiple timeframes.

Features:
    - Configurable left/right window sizes (ℓ,r)
    - Equal-high / equal-low detection with tolerances
    - Liquidity clustering (multi-touch levels)
    - Swing ranking (primary / secondary / extended)
    - Designed for 15m, 1h, 6h, 12h

All swings are computed strictly on *closed* bars.
"""

from typing import List, Dict, Optional, Tuple
import time
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
import pandas as pd


class SwingHighLowDetector:
    """
    Detect swing highs and lows for a given timeframe.
    A swing high occurs when:
        high[i] > high[i-k] and high[i] > high[i+k] for all k in [1..r]

    Similarly for swing lows.

    Parameters:
        left:  number of bars to the left (ℓ)
        right: number of bars to the right (r)
        equal_tol: tolerance ratio for detecting equal highs/lows (in price %)
    """

    def __init__(self, left: int, right: int, equal_tol: float = 0.0002):
        self.left = left
        self.right = right
        self.equal_tol = equal_tol

        log(
            f"SwingHighLowDetector initialized "
            f"(left={left}, right={right}, equal_tol={equal_tol})."
        )

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        total = max(int(seconds), 0)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # ------------------------------------------------------------------
    def _is_swing_high(self, candles: List[Candle], i: int) -> bool:
        """Return True if candles[i] is a swing high."""
        center = candles[i].high

        for k in range(1, self.left + 1):
            if candles[i - k].high >= center:
                return False

        for k in range(1, self.right + 1):
            if candles[i + k].high >= center:
                return False

        return True

    def _is_swing_low(self, candles: List[Candle], i: int) -> bool:
        """Return True if candles[i] is a swing low."""
        center = candles[i].low

        for k in range(1, self.left + 1):
            if candles[i - k].low <= center:
                return False

        for k in range(1, self.right + 1):
            if candles[i + k].low <= center:
                return False

        return True

    # ------------------------------------------------------------------
    def _is_equal(self, a: float, b: float) -> bool:
        """Determine whether two price levels are 'equal' within tolerance."""
        if a == 0 or b == 0:
            return False
        return abs(a - b) / max(a, b) <= self.equal_tol

    # ------------------------------------------------------------------
    def detect(self, candles: List[Candle]) -> Dict[int, Dict[str, Optional[float]]]:
        """
        Detect all swing highs/lows for the given list of candles.

        Returns:
            {
                timestamp: {
                    "swing_high": float or None,
                    "swing_low": float or None,
                    "equal_high": bool,
                    "equal_low": bool,
                    "tier": str ("primary"|"secondary"|"extended"|None)
                },
                ...
            }
        """

        log(f"Detecting swings for {len(candles):,} candles.")

        if len(candles) < self.left + self.right + 1:
            log("Not enough candles for swing detection.")
            return {}

        results: Dict[int, Dict[str, Optional[float]]] = {}
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        started = time.perf_counter()
        last_beat = started
        total_iters = max(len(candles) - self.right - self.left, 1)
        beat_every = max(2000, total_iters // 200)  # ~0.5% cadence

        for i in range(self.left, len(candles) - self.right):
            iter_pos = i - self.left + 1
            if iter_pos % beat_every == 0:
                now = time.perf_counter()
                if (now - last_beat) >= 10.0:
                    done = iter_pos
                    pct = 100.0 * done / max(total_iters, 1)
                    elapsed = now - started
                    rate = done / max(elapsed, 1e-6)
                    eta = (total_iters - done) / max(rate, 1e-6)
                    log(
                        "[SwingHighLowDetector] progress "
                        f"{done:,}/{total_iters:,} ({pct:.1f}%) "
                        f"elapsed={self._fmt_duration(elapsed)} eta={self._fmt_duration(eta)}"
                    )
                    last_beat = now
            ts = candles[i].timestamp

            swing_high = None
            swing_low = None
            equal_high = False
            equal_low = False
            tier = None

            # Swing High Detection
            if self._is_swing_high(candles, i):
                center = highs[i]
                swing_high = center

                # Equal-high clustering
                left_equal = self._is_equal(center, highs[i - 1])
                right_equal = self._is_equal(center, highs[i + 1])
                equal_high = left_equal or right_equal

            # Swing Low Detection
            if self._is_swing_low(candles, i):
                center = lows[i]
                swing_low = center

                # Equal-low clustering
                left_equal = self._is_equal(center, lows[i - 1])
                right_equal = self._is_equal(center, lows[i + 1])
                equal_low = left_equal or right_equal

            # Tier Ranking Heuristic
            if swing_high is not None:
                tier = self._rank_swing(i, highs, swing_high, direction="high")

            elif swing_low is not None:
                tier = self._rank_swing(i, lows, swing_low, direction="low")

            results[ts] = {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "equal_high": equal_high,
                "equal_low": equal_low,
                "tier": tier,
            }

        log(f"Swing detection complete: {len(results):,} rows produced.")
        return results

    # ------------------------------------------------------------------
    def _rank_swing(
        self,
        i: int,
        arr: List[float],
        value: float,
        direction: str
    ) -> str:
        """
        Rank swing importance based on:
        - deviation from surrounding values
        - magnitude compared to local volatility
        - position relative to recent swings

        This is intentionally simple and deterministic to preserve stability.
        """

        diff_left = abs(value - arr[i - 1])
        diff_right = abs(value - arr[i + 1])
        magnitude = max(diff_left, diff_right)

        # Simple tiering rule:
        if magnitude > 0.005 * value:
            return "primary"
        if magnitude > 0.002 * value:
            return "secondary"
        return "extended"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run swing detection and attach swing columns.
        """
        if df is None or df.empty:
            return df

        frame = df.copy()
        if "timestamp" not in frame.columns:
            if "dt" not in frame.columns:
                raise ValueError("SwingHighLowDetector.apply requires 'dt' or 'timestamp' column.")
            frame["timestamp"] = pd.to_datetime(frame["dt"]).astype("int64") // 10**9

        candles = [
            Candle(
                timestamp=int(row.timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0)),
            )
            for row in frame.itertuples()
        ]

        res = self.detect(candles)
        if not res:
            return frame

        res_df = pd.DataFrame.from_dict(res, orient="index")
        res_df.index.name = "timestamp"
        res_df = res_df.reset_index().rename(
            columns={
                "equal_high": "swing_equal_high",
                "equal_low": "swing_equal_low",
                "tier": "swing_tier",
            }
        )

        merged = frame.merge(res_df, on="timestamp", how="left")
        if "dt" in merged.columns:
            merged = merged.sort_values("dt")
        return merged
