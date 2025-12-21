"""
Pandas-based timeframe resampler for OHLCV data.
"""

import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("resampler")


class TimeframeResampler:
    """
    Simple OHLCV resampler using pandas.Grouper frequency strings (e.g., '15min', '1h').
    Expects input with columns: ['timestamp' or 'dt', 'open','high','low','close','volume'].
    """

    def __init__(self):
        LOG.info("TimeframeResampler initialized.")

    def resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        if "dt" not in df.columns:
            if "timestamp" in df.columns:
                df = df.copy()
                df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
            else:
                raise ValueError("DataFrame must contain 'dt' or 'timestamp' column.")

        grouped = (
            df.set_index("dt")
            .resample(freq, label="right", closed="right")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
            .reset_index()
        )

        grouped["timestamp"] = grouped["dt"].astype("int64") // 10**9
        return grouped[["timestamp", "dt", "open", "high", "low", "close", "volume"]]
