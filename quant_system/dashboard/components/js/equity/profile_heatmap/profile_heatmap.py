"""
profit_heatmap.py

Sends:
 - daily PnL
 - weekly summary
 - monthly summary

The JS renderer converts these into heatmap tiles.
"""

from typing import Dict, Any


class ProfitHeatmapBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def send_daily_pnl(self, ts: str, pnl: float):
        if self.js_callback:
            self.js_callback({
                "type": "heatmap_daily",
                "payload": {
                    "timestamp": ts,
                    "pnl": pnl
                }
            })

    def send_monthly_summary(self, month: str, pnl: float):
        if self.js_callback:
            self.js_callback({
                "type": "heatmap_monthly",
                "payload": {
                    "month": month,
                    "pnl": pnl
                }
            })

    def send_weekly_summary(self, week: str, pnl: float):
        if self.js_callback:
            self.js_callback({
                "type": "heatmap_weekly",
                "payload": {
                    "week": week,
                    "pnl": pnl
                }
            })
