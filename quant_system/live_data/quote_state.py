"""
quote_state.py
Maintains rolling 1m windows + latest quotes + TF buffers
for multi-asset live data.
"""

from collections import deque
from datetime import datetime
import pandas as pd


class QuoteState:
    """Per-asset live quote container."""

    def __init__(self, window_size: int = 5000):
        self.last_quote = None
        self.window_1m = deque(maxlen=window_size)

    def push_1m(self, candle: dict):
        """Store new 1m candle as dict."""
        self.last_quote = candle
        self.window_1m.append(candle)

    def get_window(self):
        """Return newest rolling window as DataFrame."""
        if not self.window_1m:
            return pd.DataFrame()
        return pd.DataFrame(self.window_1m)
