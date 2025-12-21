"""
trade_overlays.py

Bridges ForwardDashboardAdapter → JS overlay engine.
Provides:
 - entry marker push
 - stop line push
 - exit marker push
 - risk box push
 - r-multiple trail push
 - moonshot halo push
"""

from typing import Dict, Any


class TradeOverlayBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    # -----------------------------------------------------
    # SEND OVERLAY EVENT TO JS
    # -----------------------------------------------------
    def send(self, event_type: str, payload: Dict[str, Any]):
        if self.js_callback:
            self.js_callback({
                "type": event_type,
                "payload": payload
            })

    # -----------------------------------------------------
    # ENTRY
    # -----------------------------------------------------
    def push_entry(self, trade_id, ts, price, side):

        color = "#4CAF50" if side == "long" else "#E53935"

        self.send("entry_marker", {
            "trade_id": trade_id,
            "timestamp": ts,
            "price": price,
            "color": color,
            "side": side,
        })

    # -----------------------------------------------------
    # STOP LINE
    # -----------------------------------------------------
    def push_stopline(self, trade_id, stop_price):

        self.send("stop_line", {
            "trade_id": trade_id,
            "stop": stop_price,
            "color": "#FF6D00"
        })

    # -----------------------------------------------------
    # EXIT
    # -----------------------------------------------------
    def push_exit(self, trade_id, ts, price, r_mult):

        color = "#00C853" if r_mult > 0 else "#D50000"

        self.send("exit_marker", {
            "trade_id": trade_id,
            "timestamp": ts,
            "price": price,
            "r_mult": r_mult,
            "color": color
        })

    # -----------------------------------------------------
    # R-MULTIPLE TRAIL
    # -----------------------------------------------------
    def push_rtrail(self, trade_id, trail_series):

        self.send("r_trail", {
            "trade_id": trade_id,
            "series": trail_series,
            "color": "#29B6F6"
        })

    # -----------------------------------------------------
    # MOONSHOT HALO (EVR extreme)
    # -----------------------------------------------------
    def push_moonshot_halo(self, trade_id, ts, price):

        self.send("moonshot_halo", {
            "trade_id": trade_id,
            "timestamp": ts,
            "price": price,
            "glow_color": "rgba(0,180,255,0.50)",
            "radius": 24
        })
