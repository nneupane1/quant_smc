"""
Liquidity Sweep Detector
------------------------

Detects wick-based liquidity sweeps (stop runs) over prior highs/lows.

Definitions:
    Sweep-high: candle's high wicks above a prior high, then closes back below.
    Sweep-low:  candle's low wicks below a prior low, then closes back above.

Also detects equal-high/low liquidity pools using a tolerance ratio.
Tracks:
    - swept_level
    - sweep_strength (wick distance)
    - displacement_hint (close location vs bar range)
"""

from typing import List, Dict, Optional
import time
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class LiquiditySweepDetector:
    """
    Detect equal-high/low clusters and wick sweeps.
    """

    def __init__(self, equal_tol: float = 0.0002):
        self.equal_tol = equal_tol
        log(f"LiquiditySweepDetector initialized (equal_tol={equal_tol}).")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        total = max(int(seconds), 0)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _equal(self, a: float, b: float) -> bool:
        if a == 0 or b == 0:
            return False
        return abs(a - b) / max(a, b) <= self.equal_tol

    def detect(self, candles: List[Candle]) -> Dict[int, Dict[str, Optional[float]]]:
        log(f"Detecting liquidity sweeps for {len(candles):,} candles.")

        result: Dict[int, Dict[str, Optional[float]]] = {}
        if len(candles) < 3:
            log("Not enough candles for sweep detection.")
            return result

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        opens = [c.open for c in candles]
        ts_arr = [c.timestamp for c in candles]
        started = time.perf_counter()
        last_beat = started
        beat_every = max(2000, len(candles) // 200)  # ~0.5% cadence

        # Track equal-high/low liquidity pools
        eq_high = None
        eq_low = None

        for i in range(1, len(candles)):
            if i % beat_every == 0:
                now = time.perf_counter()
                if (now - last_beat) >= 10.0:
                    done = i
                    total = len(candles)
                    pct = 100.0 * done / max(total, 1)
                    elapsed = now - started
                    rate = done / max(elapsed, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    log(
                        "[LiquiditySweepDetector] progress "
                        f"{done:,}/{total:,} ({pct:.1f}%) "
                        f"elapsed={self._fmt_duration(elapsed)} eta={self._fmt_duration(eta)}"
                    )
                    last_beat = now
            ts = ts_arr[i]

            equal_high = False
            equal_low = False
            sweep_high = False
            sweep_low = False
            swept_level = None
            sweep_strength = None
            displacement_hint = None

            # Equal High Detection
            if self._equal(highs[i], highs[i - 1]):
                eq_high = highs[i]
                equal_high = True

            # Equal Low Detection
            if self._equal(lows[i], lows[i - 1]):
                eq_low = lows[i]
                equal_low = True

            # Sweep Above High
            if eq_high is not None:
                if highs[i] > eq_high and closes[i] < eq_high:
                    sweep_high = True
                    swept_level = eq_high
                    sweep_strength = highs[i] - eq_high

                    body_top = max(opens[i], closes[i])
                    bar_range = max(1e-9, highs[i] - lows[i])
                    displacement_hint = (body_top - lows[i]) / bar_range

            # Sweep Below Low
            if eq_low is not None:
                if lows[i] < eq_low and closes[i] > eq_low:
                    sweep_low = True
                    swept_level = eq_low
                    sweep_strength = eq_low - lows[i]

                    body_bottom = min(opens[i], closes[i])
                    bar_range = max(1e-9, highs[i] - lows[i])
                    displacement_hint = (highs[i] - body_bottom) / bar_range

            result[ts] = {
                "equal_high": equal_high,
                "equal_low": equal_low,
                "sweep_high": sweep_high,
                "sweep_low": sweep_low,
                "swept_level": swept_level,
                "sweep_strength": sweep_strength,
                "displacement_hint": displacement_hint
            }

        log("Liquidity sweep detection complete.")
        return result

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect sweep events and merge into dataframe.
        """
        if df is None or df.empty:
            return df

        frame = df.copy()
        if "timestamp" not in frame.columns:
            if "dt" not in frame.columns:
                raise ValueError("LiquiditySweepDetector.apply requires 'dt' or 'timestamp' column.")
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
        res_df = res_df.reset_index()
        res_df["sweep_flag"] = (
            res_df["sweep_high"].fillna(False) | res_df["sweep_low"].fillna(False)
        ).astype(int)
        res_df["sweep_dir"] = 0
        res_df.loc[res_df["sweep_high"].fillna(False), "sweep_dir"] = -1
        res_df.loc[res_df["sweep_low"].fillna(False), "sweep_dir"] = 1

        merged = frame.merge(res_df, on="timestamp", how="left")
        if "dt" in merged.columns:
            merged = merged.sort_values("dt")
        return merged
