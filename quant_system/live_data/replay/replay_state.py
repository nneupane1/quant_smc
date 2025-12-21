"""
replay_state.py
Stores multi-asset replay pointers, current bar index, and TF structures.
"""

import pandas as pd


class ReplayState:
    """Per-asset replay cursor + time-series container."""

    def __init__(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                 df_6h: pd.DataFrame, df_12h: pd.DataFrame):

        self.frames = {
            "15m": df_15m.reset_index(drop=True),
            "1h": df_1h.reset_index(drop=True),
            "6h": df_6h.reset_index(drop=True),
            "12h": df_12h.reset_index(drop=True)
        }

        self.cursor = 0   # index into 15m
        self.length = len(df_15m)

    def has_next(self) -> bool:
        return self.cursor < self.length

    def next_15m(self):
        """Return next 15m bar and increment cursor."""
        if not self.has_next():
            return None
        bar = self.frames["15m"].iloc[self.cursor].to_dict()
        self.cursor += 1
        return bar

    def get_tf_bar(self, tf: str, dt):
        """Return TF bar matching dt."""
        df = self.frames[tf]
        row = df.loc[df["dt"] == dt]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
