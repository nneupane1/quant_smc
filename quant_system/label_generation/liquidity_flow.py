"""
Liquidity-Flow Label Generator
------------------------------

Implements the sweep → displacement → continuation label used by
the Liquidity-Flow specialist model.

Definition:

    Label = 1 if AFTER a sweep (1h or 15m) and its displacement candle,
            price achieves +1R continuation BEFORE invalidation and
            within horizon H (= 12 × 15m bars).

    Label = 0 otherwise.

Inputs Required:
    candles: list[Candle] on *15m* TF (execution)
    sweeps: dict[ts → sweep info] containing:
        - "sweep_up" or "sweep_down"
        - "sweep_strength"
    displacement: dict[ts → displacement info] containing:
        - "displacement" flag
        - "body_ratio" >= 0.6
        - "vol_z" >= 0.8
    atr_15m: dict[ts → ATR]

Rules (strictly aligned to system spec):
    - Sweep occurs first.
    - Displacement must follow *immediately* (next closed 15m bar).
    - Retrace is optional (ignored for labeling but model can use features).
    - Continuation = +1R move evaluated using ATR at displacement time.
    - Invalidation occurs if price violates the opposite extreme of the sweep.
    - No lookahead; uses fully closed bars only.
"""

from typing import Dict, List, Optional
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
from quant_system.config.config_loader import ConfigLoader


class LiquidityFlowLabeler:
    """
    Generates Liquidity-Flow labels using sweep + displacement logic.

    Parameters:
        horizon_bars: default 12 = 3 hours (12 × 15m bars)
        reward_r: continuation R target (default +1R)
    """

    def __init__(self, horizon_bars: int = 12, reward_r: float = 1.0):
        self.horizon_bars = horizon_bars
        self.reward_r = reward_r
        log(
            f"LiquidityFlowLabeler initialized "
            f"(horizon_bars={horizon_bars}, reward_r={reward_r})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        sweeps: Dict[int, Dict[str, float]],
        displacement: Dict[int, Dict[str, float]],
        atr_15m: Dict[int, float]
    ) -> Dict[int, int]:
        """
        Generate binary liquidity-flow labels for each sweep.

        Returns:
            ts → 1/0 where ts = sweep timestamp.
        """

        log("Generating Liquidity-Flow labels.")

        ts_arr = [c.timestamp for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        idx = {ts: i for i, ts in enumerate(ts_arr)}

        labels: Dict[int, int] = {}

        for ts, info in sweeps.items():

            sweep_up = info.get("sweep_up", 0)
            sweep_down = info.get("sweep_down", 0)
            if not sweep_up and not sweep_down:
                continue

            if ts not in idx:
                continue

            i = idx[ts]

            # Step 1: Identify displacement candle (must be next bar)
            disp_ts = ts_arr[i + 1] if i + 1 < len(ts_arr) else None
            if disp_ts is None:
                labels[ts] = 0
                continue

            disp = displacement.get(disp_ts, {})
            if not disp.get("displacement", 0):
                labels[ts] = 0
                continue

            # Check displacement quality (body ratio >= 0.6, vol_z >= 0.8)
            if disp.get("body_ratio", 0.0) < 0.6:
                labels[ts] = 0
                continue
            if disp.get("vol_z", 0.0) < 0.8:
                labels[ts] = 0
                continue

            # Step 2: Define direction
            direction = 1 if sweep_down else -1

            # Step 3: Entry = displacement candle close
            disp_i = idx[disp_ts]
            entry_price = closes[disp_i]

            # Step 4: Stop = opposite side of sweep
            if direction == 1:  # sweep-down → long
                stop_level = lows[i]  # swept low
                target = entry_price + self.reward_r * atr_15m.get(ts, 0.0)
            else:               # sweep-up → short
                stop_level = highs[i]  # swept high
                target = entry_price - self.reward_r * atr_15m.get(ts, 0.0)

            # Step 5: Look forward H bars
            end = min(len(candles), disp_i + self.horizon_bars + 1)
            success = 0

            for j in range(disp_i + 1, end):

                # Invalidation first
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

        log(f"Liquidity-Flow label generation complete. Labels: {len(labels)}")
        return labels

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Convenience wrapper to attach label_liq_flow to a 15m dataframe.
        Uses ConfigLoader->labels.yaml if provided, otherwise defaults to ctor params.
        Expects columns: close/high/low, sweep_high/sweep_low/swept_level, atr or atr_15m.
        """
        cfg_h = self.horizon_bars
        reward_r = self.reward_r
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["liq_flow"]
            cfg_h = int(lc.get("horizon_bars", cfg_h))
            reward_r = float(lc.get("continuation_min_R", reward_r))

        df = df15.copy()
        atr_col = "atr" if "atr" in df.columns else "atr_15m" if "atr_15m" in df.columns else None
        if atr_col is None:
            df["label_liq_flow"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            direction = None
            stop = None
            if row.get("sweep_high", False):
                direction = -1
                stop = row.get("swept_level", row["high"])
            elif row.get("sweep_low", False):
                direction = 1
                stop = row.get("swept_level", row["low"])

            if direction is None or pd.isna(atr):
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

        df["label_liq_flow"] = labels
        return df
