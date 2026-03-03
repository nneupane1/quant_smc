"""Per-asset rolling quote and timeframe state."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import pandas as pd


class QuoteState:
    """Per-asset live quote container with deterministic dedupe."""

    def __init__(self, window_size: int = 5000):
        self.last_quote: Optional[dict] = None
        self.last_1m_end: Optional[int] = None
        self.window_1m: Deque[dict] = deque(maxlen=window_size)
        self.tf_windows: Dict[str, Deque[dict]] = {
            "15m": deque(maxlen=window_size),
            "1h": deque(maxlen=window_size),
            "6h": deque(maxlen=window_size),
            "12h": deque(maxlen=window_size),
        }

    def push_1m(self, candle: dict):
        ts = int(candle["timestamp"])
        if self.last_1m_end == ts:
            if self.window_1m:
                self.window_1m[-1] = candle
            self.last_quote = candle
            return
        self.last_1m_end = ts
        self.last_quote = candle
        self.window_1m.append(candle)

    def push_tf(self, tf: str, candle: dict):
        if tf not in self.tf_windows:
            self.tf_windows[tf] = deque(maxlen=self.window_1m.maxlen)
        window = self.tf_windows[tf]
        ts = int(candle["timestamp"])
        if window and int(window[-1]["timestamp"]) == ts:
            window[-1] = candle
        else:
            window.append(candle)

    def get_window(self, tf: str = "1m") -> pd.DataFrame:
        if tf == "1m":
            rows = list(self.window_1m)
        else:
            rows = list(self.tf_windows.get(tf, []))
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        if "dt" in out.columns:
            out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
            out = out.sort_values("dt")
        return out.reset_index(drop=True)

    def snapshot(self) -> dict:
        return {
            "last_quote": self.last_quote,
            "window_1m": self.get_window("1m"),
            "tf_windows": {tf: self.get_window(tf) for tf in self.tf_windows},
        }
