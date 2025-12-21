"""
dual_replay_engine.py
Master controller for dual-model replay mode:
 • Shares ReplayState + ReplayTimeline with both models
 • Steps both ForwardEngines synchronously
 • Broadcasts to dual dashboards
"""

import time
from quant_system.replay.replay_stream import ReplayStream
from quant_system.replay.replay_state import ReplayState
from quant_system.replay.replay_timeline import ReplayTimeline


class DualReplayEngine:

    def __init__(self, data, forward_A, forward_B, dashboard_A, dashboard_B):
        """
        data: { asset: { '15m':df, '1h':df, '6h':df, '12h':df } }
        forward_A: trained ForwardEngine using Model Version A
        forward_B: trained ForwardEngine using Model Version B
        dashboard_A: Streamlit dashboard adapter for Version A
        dashboard_B: Streamlit dashboard adapter for Version B
        """

        self.states = {
            asset: ReplayState(df["15m"], df["1h"], df["6h"], df["12h"])
            for asset, df in data.items()
        }

        self.timeline = ReplayTimeline(self.states)
        self.forward_A = forward_A
        self.forward_B = forward_B
        self.dashboard_A = dashboard_A
        self.dashboard_B = dashboard_B

        self.speed = 0.5
        self.running = False

    # ------------------------------------------------------------
    def start(self):
        self.running = True
        while self.running:
            self.step()
            time.sleep(self.speed)

    # ------------------------------------------------------------
    def step(self):
        stream_A = ReplayStream(self.states, self.forward_A, self.dashboard_A)
        stream_B = ReplayStream(self.states, self.forward_B, self.dashboard_B)

        stream_A.step()
        stream_B.step()

        any_left = any(st.has_next() for st in self.states.values())
        if not any_left:
            self.running = False

    # ------------------------------------------------------------
    def stop(self):
        self.running = False

    # ------------------------------------------------------------
    def set_speed(self, s):
        self.speed = max(0.01, float(s))

    # ------------------------------------------------------------
    def seek(self, idx):
        """Jump timeline for both models."""
        target_dt = self.timeline.dt_at(idx)
        if not target_dt:
            return

        # reset cursor for all assets
        for st in self.states.values():
            frame = st.frames["15m"]
            loc = frame.index[frame["dt"] == target_dt]
            if len(loc) > 0:
                st.cursor = loc[0]

        # Clear both engines before injecting replay steps
        self.forward_A.reset()
        self.forward_B.reset()

        self.dashboard_A.clear()
        self.dashboard_B.clear()
