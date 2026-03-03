"""Replay timeline helpers."""

from __future__ import annotations

import pandas as pd


class ReplayTimeline:
    """Union timeline across all replay assets."""

    def __init__(self, replay_states: dict):
        stamps = []
        for state in replay_states.values():
            df = state.frames.get("15m", pd.DataFrame())
            if df.empty:
                continue
            stamps.extend(pd.to_datetime(df["dt"], errors="coerce").dropna().tolist())
        self.timeline = sorted(set(stamps))
        self.length = len(self.timeline)

    def dt_at(self, idx: int):
        if idx < 0 or idx >= self.length:
            return None
        return self.timeline[idx]
