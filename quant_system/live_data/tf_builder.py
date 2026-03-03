"""Closed-bar 1m to higher-timeframe aggregation."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Optional

import pandas as pd


class TFBuilder:
    """Aggregate closed 1m candles into closed 15m/1h/6h/12h bars."""

    TF_MAP = {
        "15m": 15,
        "1h": 60,
        "6h": 360,
        "12h": 720,
    }

    def __init__(self):
        self.buffers = {tf: {"start": None, "end": None, "rows": []} for tf in self.TF_MAP}

    @staticmethod
    def _bucket_bounds(dt, mins: int):
        dt = pd.to_datetime(dt)
        open_ts = dt - pd.Timedelta(minutes=1)
        start = open_ts.floor(f"{mins}min")
        end = start + pd.Timedelta(minutes=mins)
        return start, end

    @staticmethod
    def _finalize(rows) -> Optional[Dict[str, object]]:
        if not rows:
            return None
        df = pd.DataFrame(rows)
        close_dt = pd.to_datetime(df["dt"].iloc[-1])
        out = {
            "dt": close_dt,
            "timestamp": int(pd.Timestamp(close_dt).timestamp()),
            "open": float(df["open"].iloc[0]),
            "high": float(df["high"].max()),
            "low": float(df["low"].min()),
            "close": float(df["close"].iloc[-1]),
            "volume": float(df["volume"].sum()),
        }
        if "asset" in df.columns:
            out["asset"] = df["asset"].iloc[-1]
        return out

    def push_1m(self, candle: dict):
        """Push one closed 1m candle and return newly closed higher-timeframe bars."""
        emits = {}
        dt = pd.to_datetime(candle["dt"])

        for tf, mins in self.TF_MAP.items():
            start, end = self._bucket_bounds(dt, mins)
            buf = self.buffers[tf]

            if buf["start"] is None:
                buf["start"], buf["end"] = start, end

            if start != buf["start"]:
                prev = self._finalize(buf["rows"])
                if prev:
                    emits[tf] = prev
                buf["rows"] = []
                buf["start"], buf["end"] = start, end

            buf["rows"].append(dict(candle))

            if dt >= buf["end"]:
                closed = self._finalize(buf["rows"])
                if closed:
                    emits[tf] = closed
                buf["rows"] = []
                buf["start"] = buf["end"]
                buf["end"] = buf["start"] + timedelta(minutes=mins)

        return emits
