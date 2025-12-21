"""
Hazard Label Generator (Discrete-Time Survival)
-----------------------------------------------

Produces time-to-failure labels for the Hazard specialist model.

Definition (from system spec):

    A trade "fails" if ANY of the following occur:
        - stop_loss is hit
        - CHOCH against position occurs
        - drawdown >= 1R occurs

    Hazard model works on discrete time bins:
        Each bin = 1 × 15m bar
        Horizon = H = 48 bins (12 hours)

Outputs:
    event_time_bin[entry_ts] = k   where k ∈ {1..H}  OR  H (censored)
    event_indicator[entry_ts] = 1 if event occurred, else 0

Notes:
    - Fully deterministic
    - Strict closed-bar logic
    - ATR determines R size dynamically at entry
    - Input: trade entries + candles + CHOCH flags + ATR
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
from quant_system.config.config_loader import ConfigLoader


class HazardLabeler:
    """
    Generate hazard survival labels.

    Parameters:
        horizon_bars: number of 15m survival bins (default 48)
        dd_r: drawdown threshold (default = 1R)
    """

    def __init__(self, horizon_bars: int = 48, dd_r: float = 1.0):
        self.horizon_bars = horizon_bars
        self.dd_r = dd_r

        log(
            f"HazardLabeler initialized "
            f"(horizon_bars={horizon_bars}, dd_r={dd_r})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        entries: Dict[int, Dict[str, float]],
        atr_15m: Dict[int, float],
        choch: Dict[int, Dict[str, float]]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Generate hazard labels.

        Parameters:
            candles: 15m candles
            entries: ts → { "direction": +1/-1, "entry_price": float }
            atr_15m: ts → ATR
            choch: ts → { "choch_up", "choch_down" } per bar

        Returns:
            event_time_bin: ts → k (1..H or H if censored)
            event_indicator: ts → 1 if event occurred, 0 otherwise
        """

        log("Generating Hazard survival labels.")

        ts_arr = [c.timestamp for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        idx = {ts: i for i, ts in enumerate(ts_arr)}
        N = len(candles)

        event_time_bin: Dict[int, int] = {}
        event_indicator: Dict[int, int] = {}

        for ts, ent in entries.items():

            if ts not in idx:
                continue

            i = idx[ts]
            entry_price = ent["entry_price"]
            direction = ent["direction"]   # +1 long, -1 short
            atr_now = atr_15m.get(ts, None)

            if atr_now is None:
                event_time_bin[ts] = self.horizon_bars
                event_indicator[ts] = 0
                continue

            # 1R price movement
            r_amount = self.dd_r * atr_now

            # Failure threshold
            if direction == 1:
                stop_level = entry_price - r_amount
            else:
                stop_level = entry_price + r_amount

            fail_bin = self.horizon_bars
            fail_event = 0

            # Scan forward survival bins
            end = min(N, i + self.horizon_bars + 1)

            for b, j in enumerate(range(i + 1, end), start=1):

                # Drawdown failure
                if direction == 1:
                    if lows[j] <= stop_level:
                        fail_bin = b
                        fail_event = 1
                        break
                else:
                    if highs[j] >= stop_level:
                        fail_bin = b
                        fail_event = 1
                        break

                # CHOCH failure against direction
                ts_j = ts_arr[j]
                ch = choch.get(ts_j, {})
                if direction == 1:
                    if ch.get("choch_down", 0):
                        fail_bin = b
                        fail_event = 1
                        break
                else:
                    if ch.get("choch_up", 0):
                        fail_bin = b
                        fail_event = 1
                        break

            event_time_bin[ts] = fail_bin
            event_indicator[ts] = fail_event

        log("Hazard label generation complete.")
        return event_time_bin, event_indicator

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Attach hazard_event and hazard_time to a 15m dataframe (downside event).
        Expects atr or atr_15m column.
        """
        H = self.horizon_bars
        dd_r = self.dd_r
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["hazard"]
            H = int(lc.get("horizon_bars", H))
            dd_r = float(lc.get("event_R_threshold", dd_r))

        df = df15.copy()
        atr_col = "atr" if "atr" in df.columns else "atr_15m" if "atr_15m" in df.columns else None
        if atr_col is None:
            df["hazard_event"] = 0
            df["hazard_time"] = H
            return df

        events = []
        times = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            if pd.isna(atr) or row["close"] <= 0:
                events.append(0)
                times.append(H)
                continue

            stop = row["close"] - dd_r * atr
            window = df.iloc[i + 1:i + 1 + H]
            event_hit = 0
            t_hit = H
            for j, (_, r) in enumerate(window.iterrows(), start=1):
                if r["low"] <= stop:
                    event_hit = 1
                    t_hit = j
                    break
            events.append(event_hit)
            times.append(t_hit)

        df["hazard_event"] = events
        df["hazard_time"] = times
        return df
