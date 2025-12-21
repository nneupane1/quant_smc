"""
cooling_panel.py

Python → JS bridge for cooling countdown visual.
Sends:
 - cooling_start timestamp
 - cooling_end timestamp
 - remaining seconds
"""

from typing import Dict, Any

class CoolingPanelBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def _send(self, event_type: str, payload: Dict[str, Any]):
        if self.js_callback:
            self.js_callback({"type": event_type, "payload": payload})

    def push_cooling_update(self, start_ts: str, end_ts: str, remaining_sec: int):
        self._send("cooling_update", {
            "cool_start": start_ts,
            "cool_end": end_ts,
            "remaining_sec": remaining_sec
        })

    def push_cooling_off(self):
        self._send("cooling_off", {})
