"""Replay state containers with timeline-safe lookup helpers."""

from __future__ import annotations

import pandas as pd


class ReplayState:
    """Per-asset replay cursor and normalized timeframe frames."""

    def __init__(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_6h: pd.DataFrame, df_12h: pd.DataFrame):
        self.frames = {
            "15m": self._normalize(df_15m),
            "1h": self._normalize(df_1h),
            "6h": self._normalize(df_6h),
            "12h": self._normalize(df_12h),
        }
        self.cursor = 0
        self.length = len(self.frames["15m"])

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "dt" not in out.columns and "timestamp" in out.columns:
            out["dt"] = pd.to_datetime(out["timestamp"], errors="coerce", unit="s")
        else:
            out["dt"] = pd.to_datetime(out["dt"], errors="coerce")
        return out.sort_values("dt").reset_index(drop=True)

    def has_next(self) -> bool:
        return self.cursor < self.length

    def peek_dt(self):
        if not self.has_next():
            return None
        return self.frames["15m"].iloc[self.cursor]["dt"]

    def next_15m(self):
        if not self.has_next():
            return None
        bar = self.frames["15m"].iloc[self.cursor].to_dict()
        self.cursor += 1
        return bar

    def seek_to(self, dt) -> None:
        if self.frames["15m"].empty:
            self.cursor = 0
            return
        dt = pd.to_datetime(dt, errors="coerce")
        idx = self.frames["15m"]["dt"].searchsorted(dt, side="left")
        self.cursor = min(int(idx), self.length)

    def get_tf_bar(self, tf: str, dt):
        df = self.frames.get(tf, pd.DataFrame())
        if df.empty:
            return None
        dt = pd.to_datetime(dt, errors="coerce")
        rows = df.loc[df["dt"] <= dt]
        if rows.empty:
            return None
        return rows.iloc[-1].to_dict()
