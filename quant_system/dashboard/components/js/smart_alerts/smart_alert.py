"""
smart_alerts.py

Smart Alerts Sidebar:
 - Confluence decomposition
 - EVR logic tree
 - Hazard snapshot
 - Gate approvals (10h/6h/1h/15m)
 - SMC structure
 - Session/regime slice
"""

from typing import Dict, Any


class SmartAlertsBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def push_alert(self, payload: Dict[str, Any]):
        """
        Payload structure:
        {
            "timestamp": "...",
            "side": "long/short",
            "symbol": "BTC/USDT",
            "confluence": {...},
            "evr": {...},
            "hazard": {...},
            "gates": {...},
            "smc": {...},
            "session": "...",
            "regime": {...}
        }
        """
        if self.js_callback:
            self.js_callback({
                "type": "smart_alert",
                "payload": payload
            })
