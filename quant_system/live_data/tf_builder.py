"""
tf_builder.py
Aggregates 1m → 15m → 1h → 6h → 12h using [start,end) close logic.
Emits a TF bar only when the window is closed.
"""

import pandas as pd
from datetime import timedelta


class TFBuilder:
    """Time-based multi-TF aggregator for live 1m streams."""

    TF_MAP = {
        "15m": 15,
        "1h": 60,
        "6h": 360,
        "12h": 720,
    }

    def __init__(self):
        # Per TF: {"start": datetime, "rows": []}
        self.buffers = {tf: {"start": None, "rows": []} for tf in self.TF_MAP}

    def _finalize(self, tf: str):
        buf = self.buffers[tf]
        rows = buf["rows"]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        close_dt = df["dt"].iloc[-1]
        tf_bar = {
            "dt": close_dt,
            "timestamp": int(pd.Timestamp(close_dt).timestamp()),
            "open": df["open"].iloc[0],
            "high": df["high"].max(),
            "low": df["low"].min(),
            "close": df["close"].iloc[-1],
            "volume": df["volume"].sum(),
        }
        asset = df["asset"].iloc[0] if "asset" in df.columns else None
        if asset is not None:
            tf_bar["asset"] = asset
        # reset
        self.buffers[tf] = {"start": None, "rows": []}
        return tf_bar

    def push_1m(self, candle: dict):
        """
        Route 1m candle into buffers & emit closed TF bars.
        Candle must include "dt" (datetime-like) and may include "asset".
        """
        emit = {}
        dt = pd.to_datetime(candle["dt"])

        for tf, mins in self.TF_MAP.items():
            buf = self.buffers[tf]
            dur = timedelta(minutes=mins)

            # Initialize bucket
            if buf["start"] is None:
                buf["start"] = dt.floor(f"{mins}min")

            # If the incoming bar belongs to a new bucket, finalize previous buckets
            while dt >= buf["start"] + dur:
                tf_bar = self._finalize(tf)
                if tf_bar:
                    emit[tf] = tf_bar
                buf["start"] += dur

            buf["rows"].append(candle)

            # Close bucket exactly at end boundary (dt is the close of the bucket)
            if dt >= buf["start"] + dur - timedelta(minutes=1):
                tf_bar = self._finalize(tf)
                if tf_bar:
                    emit[tf] = tf_bar
                    buf["start"] = dt.floor(f"{mins}min") + dur

        return emit
