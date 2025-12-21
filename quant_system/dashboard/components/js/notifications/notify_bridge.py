"""
notify_bridge.py

Sends push-worthy alerts to the frontend:
 - Tier A+ and A alerts
 - Cooling end
 - Hedge engaged/disengaged
 - Large EVR opportunity
 - Drawdown warning
"""

from typing import Dict, Any


class NotificationBridge:

    def __init__(self):
        self.js_callback = None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def send_notification(self, title: str, body: str, level: str = "info"):
        """
        level: info | warning | danger
        """
        if self.js_callback:
            self.js_callback({
                "type": "notify",
                "payload": {
                    "title": title,
                    "body": body,
                    "level": level
                }
            })
