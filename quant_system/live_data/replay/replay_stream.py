"""Replay streamer that advances assets on a synchronized union timeline."""

from __future__ import annotations

import pandas as pd


class ReplayStream:
    """Sequential 15m bar emitter for replay."""

    def __init__(self, replay_states: dict, forward_engine, dashboard_adapter):
        self.states = replay_states
        self.forward = forward_engine
        self.dashboard = dashboard_adapter

    def step_to(self, dt):
        dt = pd.to_datetime(dt, errors="coerce")
        emitted = []
        for asset, state in self.states.items():
            next_dt = state.peek_dt()
            if next_dt is None or pd.to_datetime(next_dt, errors="coerce") != dt:
                continue

            bar_15m = state.next_15m()
            bar_15m["asset"] = asset
            if self.dashboard:
                self.dashboard.update_candles({asset: bar_15m})

            self.forward.on_bar(asset, bar_15m)
            emitted.append((asset, bar_15m))

            for tf in ["1h", "6h", "12h"]:
                tf_bar = state.get_tf_bar(tf, dt)
                if tf_bar and self.dashboard and hasattr(self.dashboard, "update_tf_bar"):
                    tf_bar["asset"] = asset
                    self.dashboard.update_tf_bar(asset, tf, tf_bar)

        return emitted
