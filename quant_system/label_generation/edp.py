"""
EDP Label Generator
-------------------

Expected Drawdown Probability (EDP):

    Label = 1 if the *forward minimum price* over the next
             H = 96 × 15m bars represents a drawdown >= dd_r * ATR,
             relative to the current close.

    Label = 0 otherwise.

This is a *direction-agnostic, downside-only* metric.

Use cases:
    - EDP specialist model
    - CVaR-MPC risk posture adjustment
    - Meta-model early-warning signal
"""

from typing import Dict, List, Optional
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
from quant_system.config.config_loader import ConfigLoader


class EDPLabeler:
    """
    Generate EDP labels.

    Parameters:
        horizon_bars: lookahead window (default = 96 → 24h)
        dd_r: drawdown threshold in ATR multiples (default = 3.0R)
    """

    def __init__(self, horizon_bars: int = 96, dd_r: float = 3.0):
        self.horizon_bars = horizon_bars
        self.dd_r = dd_r

        log(
            f"EDPLabeler initialized "
            f"(horizon_bars={horizon_bars}, dd_r={dd_r})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        atr_15m: Dict[int, float]
    ) -> Dict[int, int]:
        """
        Generate EDP labels.

        Returns:
            ts → 0/1 for each 15m candle timestamp.
        """

        log("Generating EDP labels.")

        ts_arr = [c.timestamp for c in candles]
        closes = [c.close for c in candles]
        lows = [c.low for c in candles]
        idx = {ts: i for i, ts in enumerate(ts_arr)}

        N = len(candles)
        labels: Dict[int, int] = {}

        for i, ts in enumerate(ts_arr):

            close_now = closes[i]
            atr_now = atr_15m.get(ts, None)

            if atr_now is None or close_now <= 0:
                labels[ts] = 0
                continue

            # Drawdown threshold = R-multiple in price units
            dd_threshold = self.dd_r * atr_now
            target_price = close_now - dd_threshold

            end = min(N, i + self.horizon_bars + 1)
            breached = 0

            # Scan forward for minimum price
            for j in range(i + 1, end):

                # If low breaches threshold → event = 1
                if lows[j] <= target_price:
                    breached = 1
                    break

            labels[ts] = breached

        log(f"EDP label generation complete. Labels: {len(labels)}")
        return labels

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Attach label_edp to a 15m dataframe based on forward drawdown exceeding threshold.
        Expects atr or atr_15m column.
        """
        H = self.horizon_bars
        dd_r = self.dd_r
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["edp"]
            H = int(lc.get("horizon_bars", H))
            dd_r = float(lc.get("drawdown_R_threshold", dd_r))

        df = df15.copy()
        atr_col = "atr" if "atr" in df.columns else "atr_15m" if "atr_15m" in df.columns else None
        if atr_col is None:
            df["label_edp"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            if pd.isna(atr) or row["close"] <= 0:
                labels.append(0)
                continue

            target_price = row["close"] - dd_r * atr
            window = df.iloc[i + 1:i + 1 + H]
            hit = any(window["low"] <= target_price) if not window.empty else False
            labels.append(int(hit))
        df["label_edp"] = labels
        return df
