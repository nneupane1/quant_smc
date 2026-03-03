"""
Synchronous dual replay driver for side-by-side model comparisons.

This stays intentionally thin: it advances two forward/live-style engines with
the same 15m bar sequence and exposes basic play/seek controls for the dashboard.
"""

from __future__ import annotations

import time
from typing import Dict

import pandas as pd


class DualReplayEngine:
    def __init__(self, data, forward_A, forward_B, dashboard_A=None, dashboard_B=None):
        self.data = data
        self.forward_A = forward_A
        self.forward_B = forward_B
        self.dashboard_A = dashboard_A
        self.dashboard_B = dashboard_B
        self.speed = 0.5
        self.running = False
        self.cursor = 0
        self.timeline = self._build_timeline(data)

    @staticmethod
    def _build_timeline(data: Dict[str, Dict[str, pd.DataFrame]]):
        stamps = []
        for frames in data.values():
            df = frames.get("15m", pd.DataFrame())
            if df.empty:
                continue
            ts_col = "dt" if "dt" in df.columns else "timestamp"
            stamps.extend(pd.to_datetime(df[ts_col], errors="coerce").dropna().tolist())
        return sorted(set(stamps))

    def set_speed(self, seconds_per_bar: float):
        self.speed = max(float(seconds_per_bar), 0.01)

    def start(self):
        self.running = True
        while self.running and self.cursor < len(self.timeline):
            self.step()
            time.sleep(self.speed)

    def stop(self):
        self.running = False

    def seek(self, idx: int):
        self.cursor = max(0, min(int(idx), max(len(self.timeline) - 1, 0)))

    def step(self):
        if self.cursor >= len(self.timeline):
            self.running = False
            return None

        ts = self.timeline[self.cursor]
        for asset, frames in self.data.items():
            df = frames.get("15m", pd.DataFrame())
            if df.empty:
                continue
            ts_col = "dt" if "dt" in df.columns else "timestamp"
            row = df.loc[pd.to_datetime(df[ts_col], errors="coerce") == ts]
            if row.empty:
                continue
            bar = row.iloc[0].to_dict()
            if hasattr(self.forward_A, "on_bar"):
                self.forward_A.on_bar(asset, bar)
            if hasattr(self.forward_B, "on_bar"):
                self.forward_B.on_bar(asset, bar)
            if self.dashboard_A and hasattr(self.dashboard_A, "update_candles"):
                self.dashboard_A.update_candles({asset: bar})
            if self.dashboard_B and hasattr(self.dashboard_B, "update_candles"):
                self.dashboard_B.update_candles({asset: bar})

        self.cursor += 1
        if self.cursor >= len(self.timeline):
            self.running = False
        return ts
