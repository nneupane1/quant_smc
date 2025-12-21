"""
hedge_overlay.py

Sends hedge simulation visual events to the TradingView overlay engine:
 - hedge ratio line updates
 - hedge activation markers
 - net exposure series
"""

from typing import Dict, Any


class HedgeOverlayBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def _send(self, event_type: str, payload: Dict[str, Any]):
        if self.js_callback:
            self.js_callback({
                "type": event_type,
                "payload": payload
            })

    # ---------------------------------------------------------
    # RATIO LINE (continuous)
    # ---------------------------------------------------------
    def push_hedge_ratio(self, ts: str, ratio: float):
        self._send("hedge_ratio_point", {
            "timestamp": ts,
            "ratio": ratio
        })

    # ---------------------------------------------------------
    # ACTIVATION / SCALE EVENTS
    # ---------------------------------------------------------
    def push_activation(self, ts: str, action: str, ratio: float):
        self._send("hedge_activation", {
            "timestamp": ts,
            "action": action,
            "ratio": ratio
        })

    # ---------------------------------------------------------
    # NET EXPOSURE SERIES
    # ---------------------------------------------------------
    def push_net_exposure(self, ts: str, exposure: float):
        self._send("hedge_exposure_point", {
            "timestamp": ts,
            "exposure": exposure
        })
