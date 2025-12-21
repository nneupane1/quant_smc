"""
replay_stream.py
Feeds sequential bars into ForwardEngine exactly like live.
"""

from datetime import datetime


class ReplayStream:
    """Sequential bar emitter for multi-asset replay."""

    def __init__(self, replay_states: dict, forward_engine, dashboard_adapter):
        self.states = replay_states
        self.forward = forward_engine
        self.dashboard = dashboard_adapter

    def step(self):
        """
        For each asset:
          - Emit next 15m close
          - Emit corresponding 1h/6h/12h bars via dashboard
          - Drive ForwardEngine.on_bar
        """
        for asset, state in self.states.items():
            if not state.has_next():
                continue

            bar_15m = state.next_15m()
            bar_15m["asset"] = asset
            dt = bar_15m["dt"]

            # Update dashboard OHLC
            if self.dashboard:
                self.dashboard.update_candles({asset: bar_15m})

            # Push to ForwardEngine
            self.forward.on_bar(asset, bar_15m)

            # Send higher-TF bars to dashboard if they share this dt
            for tf in ["1h", "6h", "12h"]:
                tf_bar = state.get_tf_bar(tf, dt)
                if tf_bar and self.dashboard:
                    tf_bar["asset"] = asset
                    self.dashboard.update_tf_bar(asset, tf, tf_bar)
