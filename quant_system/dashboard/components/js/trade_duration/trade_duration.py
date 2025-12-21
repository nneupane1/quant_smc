"""
trade_duration.py

Sends duration data for open & closed trades to JS:
 - bar segments
 - duration color gradient
 - entry/exit alignment
"""

from typing import Dict, Any


class TradeDurationBridge:

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
    # SEND OPEN TRADE DURATION UPDATE
    # ---------------------------------------------------------
    def push_open_duration(self, trade_id: str, start_ts: str, now_ts: str, bars: int):
        self._send("duration_open", {
            "trade_id": trade_id,
            "start": start_ts,
            "now": now_ts,
            "bars": bars
        })

    # ---------------------------------------------------------
    # SEND CLOSED TRADE DURATION
    # ---------------------------------------------------------
    def push_closed_duration(self, trade_id: str, start_ts: str, end_ts: str, bars: int, r_mult: float):
        self._send("duration_closed", {
            "trade_id": trade_id,
            "start": start_ts,
            "end": end_ts,
            "bars": bars,
            "r_mult": r_mult
        })
