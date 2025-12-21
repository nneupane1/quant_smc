"""
BOS Continuation Label Generator
--------------------------------

Defines the binary label for the BOS continuation specialist model:

    Label = 1 if after a structural BOS (from 1h/6h/12h), price reaches ≥ +3R
             BEFORE invalidation (stop/CHOCH reversal) and within horizon H
             (default 48 × 15m bars).

    Label = 0 otherwise.

Core Inputs:
    candles: list[Candle] on 15m TF
    smc: dict[ts → dict], containing:
        - "bos_up": 1 if BOS up occurred at this timestamp
        - "bos_down": equivalent bearish
        - "bos_source_tf": which TF emitted the BOS (1h/6h/12h)
        - "invalidation_level": price level that would invalidate BOS
    atr: dict[ts → float], ATR_15m (for dynamic R sizing)

Rules:
    - Stop = invalidation_level
    - Reward = +3R target = entry_price + sign * (3R * atr)
    - Invalidation occurs if price crosses stop-level BEFORE target
    - Success must occur within H bars (no extension)
    - Only closed bars may be used

Output:
    ts → 0/1 labels for timestamps where BOS occurs
"""

from typing import Dict, List, Optional
from quant_system.config.config_loader import ConfigLoader
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class BOSContinuationLabeler:
    """
    Generates BOS continuation labels using closed 15m candles.

    Parameters:
        horizon_bars: forecast horizon (e.g., 48 = 12 hours)
        reward_r: minimum R-multiple required to classify success
    """

    def __init__(self, horizon_bars: int = 48, reward_r: float = 3.0):
        self.horizon_bars = horizon_bars
        self.reward_r = reward_r
        log(
            f"BOSContinuationLabeler initialized "
            f"(horizon_bars={horizon_bars}, reward_r={reward_r})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        smc: Dict[int, Dict[str, float]],
        atr: Dict[int, float]
    ) -> Dict[int, int]:
        """
        Generate labels:
            returns ts → 1 or 0 for timestamps where BOS is detected.
        """

        log("Generating BOS continuation labels.")

        ts_arr = [c.timestamp for c in candles]
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Build a lookup from timestamp → index
        idx = {ts: i for i, ts in enumerate(ts_arr)}

        labels: Dict[int, int] = {}

        for ts, info in smc.items():
            bos_up = info.get("bos_up", 0)
            bos_down = info.get("bos_down", 0)

            if not bos_up and not bos_down:
                continue

            if ts not in idx:
                continue

            i = idx[ts]
            atr_i = atr.get(ts, None)
            if atr_i is None:
                labels[ts] = 0
                continue

            entry_price = closes[i]
            invalid = info.get("invalidation_level", None)
            if invalid is None:
                labels[ts] = 0
                continue

            # Direction
            if bos_up:
                direction = 1
            else:
                direction = -1

            # Stop = invalidation level (from higher TF structure)
            stop_level = invalid

            # Target = entry + sign * (reward_r * ATR)
            target = entry_price + direction * (self.reward_r * atr_i)

            # Look ahead H bars
            end = min(len(candles), i + self.horizon_bars + 1)
            success = 0

            for j in range(i + 1, end):
                # Check invalidation first (strict ordering)
                if direction == 1:
                    if lows[j] <= stop_level:
                        success = 0
                        break
                    if highs[j] >= target:
                        success = 1
                        break
                else:
                    if highs[j] >= stop_level:
                        success = 0
                        break
                    if lows[j] <= target:
                        success = 1
                        break

            labels[ts] = success

        log(f"BOS continuation label generation complete. Labels: {len(labels)}")
        return labels

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Attach label_bos_cont to a 15m dataframe.
        Expects columns: bos_up/bos_down/broken_level, close/high/low, atr or atr_15m.
        """
        cfg_h = self.horizon_bars
        reward_r = self.reward_r
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["bos_cont"]
            cfg_h = int(lc.get("horizon_bars", cfg_h))
            reward_r = float(lc.get("min_R", reward_r))

        df = df15.copy()
        atr_col = "atr" if "atr" in df.columns else "atr_15m" if "atr_15m" in df.columns else None
        if atr_col is None:
            df["label_bos_cont"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            direction = None
            stop = row.get("broken_level", None)
            if row.get("bos_up", False):
                direction = 1
            elif row.get("bos_down", False):
                direction = -1
            if direction is None or stop is None or pd.isna(atr):
                labels.append(0)
                continue

            entry = row["close"]
            target = entry + direction * reward_r * atr
            window = df.iloc[i + 1:i + 1 + cfg_h]
            success = 0
            for _, r in window.iterrows():
                if direction == 1:
                    if r["low"] <= stop:
                        break
                    if r["high"] >= target:
                        success = 1
                        break
                else:
                    if r["high"] >= stop:
                        break
                    if r["low"] <= target:
                        success = 1
                        break
            labels.append(success)

        df["label_bos_cont"] = labels
        return df
