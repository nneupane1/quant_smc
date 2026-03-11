"""
BOS / CHOCH Detector
--------------------

Detects:
- Break of Structure (BOS): continuation signal after taking a prior swing
- Change of Character (CHOCH): structural reversal signal

Rules are strictly non-repainting and rely only on closed candles.
ATR-smoothed buffers can be applied in upper layers.

Inputs:
- List[Candle]
- Detected swings (optional, but recommended)

Outputs:
    {
        timestamp: {
            "bos_up": bool,
            "bos_down": bool,
            "choch_up": bool,
            "choch_down": bool,
            "broken_level": float or None
        },
        ...
    }
"""

from typing import List, Dict, Optional
import time
import pandas as pd

from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class BOSCHOCHDetector:
    """
    Detect BOS and CHOCH using swing structure.

    Parameters:
        atr_buffer: float, optional multiplier for thresholding structure breaks
    """

    def __init__(self, atr_buffer: float = 0.0):
        self.atr_buffer = atr_buffer
        log(f"BOSCHOCHDetector initialized (atr_buffer={atr_buffer}).")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        total = max(int(seconds), 0)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def detect(
        self,
        candles: List[Candle],
        swings: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, Optional[float]]]:
        """
        Detect BOS and CHOCH events and return a dict keyed by timestamp.

        candles: list of closed candles
        swings: dictionary indexed by timestamp from SwingHighLowDetector
        """

        log(f"Detecting BOS/CHOCH for {len(candles):,} candles.")

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        ts_arr = [c.timestamp for c in candles]

        result: Dict[int, Dict[str, Optional[float]]] = {}
        started = time.perf_counter()
        last_beat = started
        beat_every = max(2000, len(candles) // 200)  # ~0.5% cadence

        last_swing_high = None
        last_swing_low = None

        for i, candle in enumerate(candles):
            if i > 0 and i % beat_every == 0:
                now = time.perf_counter()
                if (now - last_beat) >= 10.0:
                    done = i
                    total = len(candles)
                    pct = 100.0 * done / max(total, 1)
                    elapsed = now - started
                    rate = done / max(elapsed, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    log(
                        "[BOSCHOCHDetector] progress "
                        f"{done:,}/{total:,} ({pct:.1f}%) "
                        f"elapsed={self._fmt_duration(elapsed)} eta={self._fmt_duration(eta)}"
                    )
                    last_beat = now
            ts = candle.timestamp
            sh = swings.get(ts, {}).get("swing_high")
            sl = swings.get(ts, {}).get("swing_low")

            # Track last valid (non-NaN) swing levels
            if sh is not None and pd.notna(sh):
                last_swing_high = float(sh)
            if sl is not None and pd.notna(sl):
                last_swing_low = float(sl)

            # Initialize outputs
            bos_up = False
            bos_down = False
            choch_up = False
            choch_down = False
            broken = None

            # BOS Up (price closes beyond last swing high)
            if last_swing_high is not None and candle.close > last_swing_high + self.atr_buffer:
                bos_up = True
                broken = last_swing_high

            # BOS Down (price closes below last swing low)
            if last_swing_low is not None and candle.close < last_swing_low - self.atr_buffer:
                bos_down = True
                broken = last_swing_low

            # CHOCH detection
            # CHOCH is recognized when direction of the BOS shifts compared to prior structure.
            # The rules below follow a simple, deterministic approach.
            if bos_up and last_swing_low is not None:
                # Up BOS following a down structure -> CHOCH to upside
                if candle.close > highs[max(0, i - 1)]:
                    choch_up = True

            if bos_down and last_swing_high is not None:
                # Down BOS following an up structure -> CHOCH to downside
                if candle.close < lows[max(0, i - 1)]:
                    choch_down = True

            result[ts] = {
                "bos_up": bos_up,
                "bos_down": bos_down,
                "choch_up": choch_up,
                "choch_down": choch_down,
                "broken_level": broken,
            }

        log("BOS/CHOCH detection complete.")
        return result

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect BOS/CHOCH events and merge into dataframe.
        """
        if df is None or df.empty:
            return df

        frame = df.copy()
        if "timestamp" not in frame.columns:
            if "dt" not in frame.columns:
                raise ValueError("BOSCHOCHDetector.apply requires 'dt' or 'timestamp' column.")
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

        # Build swings map from existing columns if present; otherwise fallback detection.
        swings = {}
        if "swing_high" in frame.columns or "swing_low" in frame.columns:
            for row in frame.itertuples():
                sh = getattr(row, "swing_high", None)
                sl = getattr(row, "swing_low", None)
                swings[int(row.timestamp)] = {
                    "swing_high": None if pd.isna(sh) else float(sh),
                    "swing_low": None if pd.isna(sl) else float(sl),
                }
        else:
            from quant_system.features.smc.swings import SwingHighLowDetector

            swing_det = SwingHighLowDetector(left=2, right=2)
            swings = swing_det.detect(candles)

        res = self.detect(candles, swings)
        if not res:
            return frame

        res_df = pd.DataFrame.from_dict(res, orient="index")
        res_df.index.name = "timestamp"
        res_df = res_df.reset_index()
        res_df["bos_flag"] = (res_df["bos_up"].fillna(False) | res_df["bos_down"].fillna(False)).astype(int)
        res_df["choch_flag"] = (res_df["choch_up"].fillna(False) | res_df["choch_down"].fillna(False)).astype(int)

        bias = pd.Series("NEUTRAL", index=res_df.index, dtype="object")
        bias = bias.mask(res_df["bos_up"].fillna(False), "UP")
        bias = bias.mask(res_df["bos_down"].fillna(False), "DOWN")
        bias = bias.mask(res_df["choch_up"].fillna(False), "UP")
        bias = bias.mask(res_df["choch_down"].fillna(False), "DOWN")
        res_df["bias"] = bias
        res_df["structure_bias"] = bias

        merged = frame.merge(res_df, on="timestamp", how="left")
        if "dt" in merged.columns:
            merged = merged.sort_values("dt")
        return merged
