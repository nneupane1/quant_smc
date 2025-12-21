"""
orderflow_bridge.py
Bridge between Python (Kraken feed + volume agg) and JS renderer.
"""

from typing import Dict, Any, List


class OrderflowBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def _send(self, event_type: str, payload: Dict[str, Any]):
        if self.js_callback:
            self.js_callback({"type": event_type, "payload": payload})

    # -------------------------------------------------------
    # L2 ORDERBOOK SNAPSHOT (top N levels)
    # -------------------------------------------------------
    def push_orderbook(self, bids: List[List[float]], asks: List[List[float]]):
        """
        bids / asks: [[price, size], ...]
        """
        self._send("orderbook_update", {
            "bids": bids,
            "asks": asks
        })

    # -------------------------------------------------------
    # DELTA BARS (aggressive buying vs selling)
    # -------------------------------------------------------
    def push_delta(self, timestamp: str, buy_vol: float, sell_vol: float):
        self._send("delta_update", {
            "timestamp": timestamp,
            "buy": buy_vol,
            "sell": sell_vol,
            "delta": buy_vol - sell_vol
        })

    # -------------------------------------------------------
    # FOOTPRINT (volume at each price along the candle)
    # -------------------------------------------------------
    def push_footprint(self, candle_ts: str, levels: Dict[str, Dict[str, float]]):
        """
        levels = {
            "price1": {"bid": x, "ask": y},
            "price2": {...},
            ...
        }
        """
        self._send("footprint_update", {
            "timestamp": candle_ts,
            "levels": levels
        })
