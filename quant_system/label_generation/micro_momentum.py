"""
Micro-Momentum Label Generator
------------------------------

Implements the short-horizon acceleration/exhaustion label used by the
Micro-Momentum specialist model.

Definition (strict from spec):

    Label = 1 if forward mean return over horizon H (4–8 bars)
             exceeds a volatility-adjusted noise threshold.

    Label = 0 otherwise.

Noise threshold:
    threshold = k_noise * ATR_15m[t] / close[t]
    where:
        - k_noise ∈ [0.2, 0.5] (configurable)
        - ATR ensures the threshold adapts to volatility regimes

Horizon:
    User-configurable, default = 6 bars (middle of 4–8)

All computations strictly use CLOSED bars. No lookahead contamination.
"""

from typing import Dict, List, Optional
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
from quant_system.config.config_loader import ConfigLoader


class MicroMomentumLabeler:
    """
    Computes micro-momentum labels.

    Parameters:
        horizon_bars: default 6 (midpoint of 4–8, per spec)
        noise_k: multiplicative factor for ATR-based threshold
    """

    def __init__(self, horizon_bars: int = 6, noise_k: float = 0.3):
        self.horizon_bars = horizon_bars
        self.noise_k = noise_k
        log(
            f"MicroMomentumLabeler initialized "
            f"(horizon_bars={horizon_bars}, noise_k={noise_k})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        atr_15m: Dict[int, float]
    ) -> Dict[int, int]:
        """
        Generate micro-momentum labels.

        Returns:
            ts → {0,1}
        """

        log("Generating Micro-Momentum labels.")

        ts_arr = [c.timestamp for c in candles]
        closes = [c.close for c in candles]
        idx = {ts: i for i, ts in enumerate(ts_arr)}

        labels: Dict[int, int] = {}

        N = len(candles)

        for i, ts in enumerate(ts_arr):

            close_now = closes[i]
            atr_now = atr_15m.get(ts, None)

            if atr_now is None or close_now <= 0:
                labels[ts] = 0
                continue

            # Volatility-adjusted noise threshold
            threshold = self.noise_k * (atr_now / close_now)

            end = min(N, i + self.horizon_bars + 1)
            if end <= i + 1:
                labels[ts] = 0
                continue

            forward_returns = []

            # Compute returns for each future bar j
            for j in range(i + 1, end):
                r = (closes[j] - close_now) / close_now
                forward_returns.append(r)

            if not forward_returns:
                labels[ts] = 0
                continue

            mean_forward_return = sum(forward_returns) / len(forward_returns)

            # Positive momentum must exceed noise threshold
            label = 1 if mean_forward_return > threshold else 0
            labels[ts] = label

        log(f"Micro-Momentum label generation complete. Labels: {len(labels)}")
        return labels

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Attach label_momo using forward mean return vs ATR-based noise threshold.
        """
        h = self.horizon_bars
        noise_k = self.noise_k
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["momo"]
            h = int(lc.get("max_horizon", h))
            noise_k = float(lc.get("noise_band_sigma", noise_k))

        df = df15.copy()
        atr_col = "atr" if "atr" in df.columns else "atr_15m" if "atr_15m" in df.columns else None
        if atr_col is None:
            df["label_momo"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            if pd.isna(atr) or row["close"] <= 0:
                labels.append(0)
                continue
            window = df.iloc[i + 1:i + 1 + h]
            if window.empty:
                labels.append(0)
                continue
            mean_ret = (window["close"].mean() - row["close"]) / row["close"]
            noise_thr = noise_k * (atr / row["close"])
            labels.append(int(mean_ret >= noise_thr))
        df["label_momo"] = labels
        return df
