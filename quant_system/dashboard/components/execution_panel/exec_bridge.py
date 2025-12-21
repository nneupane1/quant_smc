"""
exec_bridge.py

Bridge for the Live Execution Panel.
User modifies:
 - leverage
 - position size
 - TP / SL levels
 - risk mode
 - hedge toggle
 - force exit

Python updates ForwardExecutor state accordingly.
"""

from typing import Callable, Dict, Any


class ExecutionBridge:

    def __init__(self):
        self.js_callback = None
        self.on_action: Callable[[Dict[str, Any]], None] = lambda payload: None

    def register_js_callback(self, cb):
        self.js_callback = cb

    def register_py_action(self, cb):
        self.on_action = cb

    # --- Frontend → Backend -------------------------------------------------
    def handle_action(self, payload: Dict[str, Any]):
        """
        payload structure:
        {
           "action": "set_sl" | "set_tp" | "set_size" | "set_leverage"
                     "toggle_hedge" | "force_exit" | "risk_mode",
           "data": {...}
        }
        """
        self.on_action(payload)

    # --- Backend → Frontend (sync updated values) ---------------------------
    def push_state(self, state: Dict[str, Any]):
        if self.js_callback:
            self.js_callback({
                "type": "exec_state",
                "payload": state
            })
