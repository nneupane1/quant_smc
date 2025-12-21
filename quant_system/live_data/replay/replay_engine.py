"""
replay_engine.py
Master replay controller that drives:
 - ReplayStream
 - ReplayTimeline
 - ForwardEngine
 - Dashboard (JS + Streamlit)

Real-time playback with speed controls and asset switching.
"""

import time
import pandas as pd
from typing import Dict

from quant_system.live_data.replay.replay_state import ReplayState
from quant_system.live_data.replay.replay_stream import ReplayStream
from quant_system.live_data.replay.replay_timeline import ReplayTimeline


class ReplayEngine:
    """Full multi-asset historical replay engine."""

    def __init__(self, data: Dict[str, Dict[str, pd.DataFrame]],
                 forward_engine,
                 dashboard_adapter):

        """
        data = {
            "BTCUSDT": {
                "15m": df15m,
                "1h": df1h,
                "6h": df6h,
                "12h": df12h
            },
            ...
        }
        """

        self.states = {
            asset: ReplayState(df["15m"], df["1h"], df["6h"], df["12h"])
            for asset, df in data.items()
        }

        self.forward = forward_engine
        self.dashboard = dashboard_adapter
        self.timeline = ReplayTimeline(self.states)

        # Playback config
        self.speed = 1.0        # seconds per bar
        self.running = False

    # ----------------------------------------------------------
    # START REPLAY
    # ----------------------------------------------------------
    def start(self):
        self.running = True

        while self.running:
            any_left = False
            for asset, st in self.states.items():
                if st.has_next():
                    any_left = True

            if not any_left:
                self.running = False
                break

            self.step()
            time.sleep(self.speed)

    # ----------------------------------------------------------
    def step(self):
        """Single replay step: emit bars for all assets."""
        if not self.running:
            return

        stream = ReplayStream(self.states, self.forward, self.dashboard)
        stream.step()

    # ----------------------------------------------------------
    def stop(self):
        self.running = False

    # ----------------------------------------------------------
    def set_speed(self, seconds_per_bar: float):
        self.speed = max(0.01, seconds_per_bar)

    # ----------------------------------------------------------
    def seek(self, idx: int):
        """Jump to timeline index."""
        target_dt = self.timeline.dt_at(idx)
        if not target_dt:
            return

        # reset pointers to the target index
        for _, st in self.states.items():
            df = st.frames["15m"]
            # find row matching dt
            row = df.index[df["dt"] == target_dt]
            if len(row) > 0:
                st.cursor = row[0]
