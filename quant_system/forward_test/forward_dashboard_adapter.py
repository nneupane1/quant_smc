"""
ForwardDashboardAdapter
Lightweight bridge between the forward engine and any dashboard (Streamlit or UI server).

Responsibilities:
 - track latest state and equity curve
 - push trades/events/reasoning to UI via registered callbacks
 - provide snapshots for polling-based dashboards
"""

from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("dashboard_adapter")


class ForwardDashboardAdapter:
    def __init__(self):
        self.last_state: Optional[Any] = None
        self.event_log: List[Dict[str, Any]] = []
        self.live_equity_df = pd.DataFrame(columns=["timestamp", "equity"])
        self.candles = pd.DataFrame()

        self.callbacks: Dict[str, Optional[Callable]] = {
            "update_equity": None,
            "update_panels": None,
            "update_events": None,
            "update_trades": None,
            "update_reasoning": None,
            "update_candles": None,
            "update_hedge": None,
            "update_smart_alerts": None,
            "notify": None,
        }

    # ------------------------------------------------------------------
    def register_callbacks(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.callbacks:
                self.callbacks[k] = v

    # ------------------------------------------------------------------
    def update_state(self, state):
        """
        Called every tick by the forward engine.
        `state` is expected to expose equity, free_capital, locked_profit, current_risk_mode, current_hedge_ratio,
        cooling_end_time, open_trades, and closed_trades.
        """
        self.last_state = state
        ts = datetime.utcnow().isoformat()

        # equity curve
        self.live_equity_df.loc[len(self.live_equity_df)] = [ts, state.equity]
        if self.callbacks["update_equity"]:
            self.callbacks["update_equity"](self.live_equity_df)

        # panel data
        panel = {
            "equity": state.equity,
            "free_capital": state.free_capital,
            "locked_profit": state.locked_profit,
            "risk_mode": getattr(state, "current_risk_mode", None),
            "hedge_ratio": getattr(state, "current_hedge_ratio", None),
            "cooling_to": state.cooling_end_time.isoformat() if getattr(state, "cooling_end_time", None) else None,
        }
        if self.callbacks["update_panels"]:
            self.callbacks["update_panels"](panel)

        # hedge/exposure streams (optional)
        if self.callbacks["update_hedge"]:
            self.callbacks["update_hedge"]({
                "type": "hedge_ratio_point",
                "payload": {"timestamp": ts, "ratio": panel["hedge_ratio"]}
            })

        # trades tables
        self._update_trades_view(state)

    # ------------------------------------------------------------------
    def update_candles(self, df: Dict[str, Any]):
        self.candles = pd.DataFrame(df)
        if self.callbacks["update_candles"]:
            self.callbacks["update_candles"](self.candles)

    # ------------------------------------------------------------------
    def log_event(self, event_type: str, trade_id: str, payload: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "trade_id": trade_id,
            "payload": payload,
        }
        self.event_log.append(entry)

        if self.callbacks["update_events"]:
            self.callbacks["update_events"](entry)
        if self.callbacks["update_reasoning"]:
            self.callbacks["update_reasoning"](entry)

    # ------------------------------------------------------------------
    def log_reasoned_signal(self, payload: Dict[str, Any]):
        """
        Shortcut for high-value alerts (e.g., tier A+ signals).
        """
        if self.callbacks["update_smart_alerts"]:
            self.callbacks["update_smart_alerts"](payload)
        self.log_event("reasoning", payload.get("trade_id"), payload)

    # ------------------------------------------------------------------
    def push_notification(self, title: str, body: str, level: str = "info"):
        if self.callbacks["notify"]:
            self.callbacks["notify"]({
                "title": title,
                "body": body,
                "level": level
            })

    # ------------------------------------------------------------------
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Provides a polling-friendly snapshot for simple dashboards.
        """
        state_dict = {}
        if self.last_state:
            state_dict = {
                "equity": getattr(self.last_state, "equity", 0),
                "free_capital": getattr(self.last_state, "free_capital", 0),
                "locked_profit": getattr(self.last_state, "locked_profit", 0),
                "open_trades": getattr(self.last_state, "open_trades", {}),
            }
        return {
            "state": state_dict,
            "events": list(self.event_log),
            "candles": self.candles,
        }

    # ------------------------------------------------------------------
    def _update_trades_view(self, state):
        open_rows = []
        for tid, t in getattr(state, "open_trades", {}).items():
            open_rows.append({
                "trade_id": tid,
                "side": getattr(t, "side", ""),
                "entry": getattr(t, "entry_price", 0.0),
                "stop": getattr(t, "stop_price", None),
                "r_mult": getattr(t, "r_mult", 0.0),
                "highest_r": getattr(t, "highest_r", 0.0),
                "size": getattr(t, "size", 0.0),
                "timestamp": getattr(t, "timestamp", datetime.utcnow()).isoformat(),
            })

        closed_rows = []
        for tid, t in getattr(state, "closed_trades", {}).items():
            row = dict(t) if isinstance(t, dict) else {"trade_id": tid, **t.__dict__}
            closed_rows.append(row)

        payload = {"open": pd.DataFrame(open_rows), "closed": pd.DataFrame(closed_rows)}

        if self.callbacks["update_trades"]:
            self.callbacks["update_trades"](payload)
