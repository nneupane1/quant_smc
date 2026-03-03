"""Historical replay driver aligned with the live forward loop."""

from __future__ import annotations

import time
from typing import Dict

import pandas as pd

from quant_system.live_data.replay.replay_state import ReplayState
from quant_system.live_data.replay.replay_stream import ReplayStream
from quant_system.live_data.replay.replay_timeline import ReplayTimeline


class ReplayEngine:
    """Multi-asset historical replay engine with manual and autoplay stepping."""

    def __init__(self, data: Dict[str, Dict[str, pd.DataFrame]], forward_engine, dashboard_adapter):
        self.states = {
            asset: ReplayState(df.get("15m", pd.DataFrame()), df.get("1h", pd.DataFrame()), df.get("6h", pd.DataFrame()), df.get("12h", pd.DataFrame()))
            for asset, df in data.items()
        }
        self.forward = forward_engine
        self.dashboard = dashboard_adapter
        self.timeline = ReplayTimeline(self.states)
        self.stream = ReplayStream(self.states, self.forward, self.dashboard)

        self.speed = 1.0
        self.running = False
        self.cursor = 0

    def start(self):
        self.running = True
        while self.running and self.cursor < self.timeline.length:
            self.step()
            time.sleep(self.speed)

    def step(self):
        if self.cursor >= self.timeline.length:
            self.running = False
            return None
        dt = self.timeline.dt_at(self.cursor)
        emitted = self.stream.step_to(dt)
        self.cursor += 1
        if self.cursor >= self.timeline.length:
            self.running = False
        return emitted

    def stop(self):
        self.running = False

    def set_speed(self, seconds_per_bar: float):
        self.speed = max(0.01, float(seconds_per_bar))

    def seek(self, idx: int):
        self.cursor = max(0, min(int(idx), max(self.timeline.length - 1, 0)))
        target_dt = self.timeline.dt_at(self.cursor)
        if target_dt is None:
            return
        for state in self.states.values():
            state.seek_to(target_dt)
