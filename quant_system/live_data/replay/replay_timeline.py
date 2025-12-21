"""
replay_timeline.py
Provides an index → datetime mapping for the dashboard's slider widget.
"""

class ReplayTimeline:
    """Maps replay indices to timestamps."""

    def __init__(self, replay_states: dict):
        # Assume all assets share same 15m timeline
        any_asset = next(iter(replay_states))
        self.timeline = replay_states[any_asset].frames["15m"]["dt"].tolist()
        self.length = len(self.timeline)

    def dt_at(self, idx: int):
        if idx < 0 or idx >= self.length:
            return None
        return self.timeline[idx]
