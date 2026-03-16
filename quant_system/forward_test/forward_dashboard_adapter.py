"""
ForwardDashboardAdapter
Lightweight bridge between the forward engine and any dashboard (Streamlit or UI server).

Responsibilities:
 - track latest state and equity curve
 - push trades/events/reasoning to UI via registered callbacks
 - provide snapshots for telemetry-fed or polling-fallback dashboards
"""

from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
import pandas as pd

from quant_system.telemetry.hub import get_telemetry_hub
from quant_system.utils.logger import get_logger

LOG = get_logger("dashboard_adapter")


class ForwardDashboardAdapter:
    def __init__(self, telemetry_hub=None):
        self.last_state: Optional[Any] = None
        self.event_log: List[Dict[str, Any]] = []
        self.live_equity_df = pd.DataFrame(columns=["timestamp", "equity"])
        self.candles = pd.DataFrame()
        self.tf_bars: Dict[str, pd.DataFrame] = {}
        self.backtest_bundle: Dict[str, Any] = {}
        self.max_candles = 2000
        self.telemetry = telemetry_hub or get_telemetry_hub()

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

    @staticmethod
    def _state_get(state, key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    @classmethod
    def _trade_get(cls, trade, key, default=None):
        if isinstance(trade, dict):
            return trade.get(key, default)
        return getattr(trade, key, default)

    @staticmethod
    def _iso_ts(value):
        if value is None:
            return datetime.utcnow().isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

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
        ts = self._iso_ts(self._state_get(state, "timestamp", datetime.utcnow()))
        equity = self._state_get(state, "equity", 0.0)
        free_capital = self._state_get(state, "free_capital", 0.0)
        locked_profit = self._state_get(state, "locked_profit", 0.0)
        cooling_to = self._state_get(state, "cooling_to", None)
        if cooling_to is None:
            cooling_end_time = self._state_get(state, "cooling_end_time", None)
            cooling_to = cooling_end_time.isoformat() if cooling_end_time is not None else None

        # equity curve
        self.live_equity_df.loc[len(self.live_equity_df)] = [ts, equity]
        self.live_equity_df = (
            self.live_equity_df
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(self.max_candles)
            .reset_index(drop=True)
        )
        if self.callbacks["update_equity"]:
            self.callbacks["update_equity"](self.live_equity_df)

        # panel data
        panel = {
            "equity": equity,
            "free_capital": free_capital,
            "locked_profit": locked_profit,
            "risk_mode": self._state_get(state, "current_risk_mode", self._state_get(state, "risk_mode", None)),
            "hedge_ratio": self._state_get(state, "current_hedge_ratio", self._state_get(state, "hedge_ratio", None)),
            "cooling_to": cooling_to,
            "confluence": self._state_get(state, "confluence", None),
            "evr": self._state_get(state, "evr", None),
            "hazard": self._state_get(state, "hazard", None),
            "flow_1h": self._state_get(state, "flow_1h", None),
            "max_drawdown": self._state_get(state, "max_drawdown", None),
            "open_positions": self._state_get(state, "open_positions", None),
            "closed_trades": self._state_get(state, "closed_trades", {}),
            "exposures": self._state_get(state, "exposures", {}),
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
        self._publish_snapshot()

    # ------------------------------------------------------------------
    def update_candles(self, df: Dict[str, Any]):
        if isinstance(df, pd.DataFrame):
            candles = df.copy()
        elif isinstance(df, dict):
            rows = []
            for asset, payload in df.items():
                if isinstance(payload, dict):
                    row = dict(payload)
                    row.setdefault("asset", asset)
                    rows.append(row)
                elif isinstance(payload, pd.Series):
                    row = payload.to_dict()
                    row.setdefault("asset", asset)
                    rows.append(row)
                elif isinstance(payload, list):
                    for item in payload:
                        row = dict(item)
                        row.setdefault("asset", asset)
                        rows.append(row)
            candles = pd.DataFrame(rows)
        else:
            candles = pd.DataFrame()
        if candles.empty:
            return
        if "dt" in candles.columns:
            candles["dt"] = pd.to_datetime(candles["dt"], errors="coerce")
        elif "timestamp" in candles.columns:
            candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
        combined = pd.concat([self.candles, candles], ignore_index=True)
        subset = [col for col in ("asset", "dt", "timestamp") if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        sort_col = "dt" if "dt" in combined.columns else ("timestamp" if "timestamp" in combined.columns else None)
        if sort_col is not None:
            combined = combined.sort_values(sort_col)
        self.candles = combined.tail(self.max_candles).reset_index(drop=True)
        if self.callbacks["update_candles"]:
            self.callbacks["update_candles"](self.candles)
        self._publish_snapshot()

    # ------------------------------------------------------------------
    def log_event(self, event_type: str, trade_id: str, payload: Dict[str, Any]):
        payload = payload or {}
        event_ts = (
            payload.get("timestamp")
            or payload.get("dt")
            or payload.get("exit_ts")
            or payload.get("entry_ts")
            or datetime.utcnow()
        )
        entry = {
            "timestamp": self._iso_ts(event_ts),
            "event_type": event_type,
            "trade_id": trade_id,
            "payload": payload,
        }
        self.event_log.append(entry)

        if self.callbacks["update_events"]:
            self.callbacks["update_events"](entry)
        if self.callbacks["update_reasoning"]:
            self.callbacks["update_reasoning"](entry)
        if self.telemetry is not None:
            try:
                self.telemetry.publish_event(entry)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Telemetry event publish failed: %s", exc)
        self._publish_snapshot()

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
            base = dict(self.last_state) if isinstance(self.last_state, dict) else {}
            state_dict = {
                **base,
                "equity": self._state_get(self.last_state, "equity", 0),
                "free_capital": self._state_get(self.last_state, "free_capital", 0),
                "locked_profit": self._state_get(self.last_state, "locked_profit", 0),
                "open_trades": self._state_get(self.last_state, "open_trades", {}),
                "closed_trades": self._state_get(self.last_state, "closed_trades", {}),
                "cooling_to": self._state_get(self.last_state, "cooling_to", None),
                "confluence": self._state_get(self.last_state, "confluence", None),
                "evr": self._state_get(self.last_state, "evr", None),
                "hazard": self._state_get(self.last_state, "hazard", None),
                "flow_1h": self._state_get(self.last_state, "flow_1h", None),
                "max_drawdown": self._state_get(self.last_state, "max_drawdown", None),
                "exposures": self._state_get(self.last_state, "exposures", {}),
            }
        return {
            "state": state_dict,
            "events": list(self.event_log),
            "candles": self.candles,
            "tf_bars": {tf: df.copy() for tf, df in self.tf_bars.items()},
            "backtest": dict(self.backtest_bundle),
        }

    # ------------------------------------------------------------------
    def update_backtest_bundle(self, result: Dict[str, Any]):
        payload = dict(result or {})
        self.backtest_bundle = {
            "summary": payload.get("metrics", payload.get("summary", {})),
            "trades": payload.get("trades", pd.DataFrame()),
            "equity_curve": payload.get("equity_curve", pd.DataFrame()),
            "execution_log": payload.get("execution_log", pd.DataFrame()),
            "candles": payload.get("candles", self.candles.copy()),
            "smc_features": payload.get("smc_features", payload.get("candles", self.candles.copy())),
            "daily": payload.get("daily", pd.DataFrame()),
            "monthly": payload.get("monthly", pd.DataFrame()),
            "reasoning": payload.get("reasoning", {}),
        }
        self._publish_snapshot()

    # ------------------------------------------------------------------
    def update_tf_bar(self, asset: str, tf: str, bar: Dict[str, Any]):
        row = dict(bar or {})
        row.setdefault("asset", asset)
        frame = pd.DataFrame([row])
        if "dt" in frame.columns:
            frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
        existing = self.tf_bars.get(tf, pd.DataFrame())
        combined = pd.concat([existing, frame], ignore_index=True)
        subset = [col for col in ("asset", "dt", "timestamp") if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        sort_col = "dt" if "dt" in combined.columns else ("timestamp" if "timestamp" in combined.columns else None)
        if sort_col is not None:
            combined = combined.sort_values(sort_col)
        self.tf_bars[tf] = combined.tail(self.max_candles).reset_index(drop=True)
        if self.callbacks["update_events"]:
            self.callbacks["update_events"](
                {
                    "timestamp": self._iso_ts(row.get("dt") or row.get("timestamp")),
                    "event_type": "tf_bar",
                    "trade_id": None,
                    "payload": {"asset": asset, "timeframe": tf, **row},
                }
            )
        self._publish_snapshot()

    # ------------------------------------------------------------------
    def _update_trades_view(self, state):
        open_rows = []
        open_trades = self._state_get(state, "open_trades", {})
        if isinstance(open_trades, dict):
            iterable = open_trades.items()
        elif isinstance(open_trades, list):
            iterable = enumerate(open_trades)
        else:
            iterable = []
        for tid, t in iterable:
            open_rows.append({
                "trade_id": tid,
                "asset": self._trade_get(t, "asset", ""),
                "leg": self._trade_get(t, "leg", self._trade_get(t, "metadata", {}).get("leg") if isinstance(self._trade_get(t, "metadata", {}), dict) else ""),
                "side": self._trade_get(t, "side", ""),
                "entry": self._trade_get(t, "entry_price", 0.0),
                "stop": self._trade_get(t, "stop_price", None),
                "r_mult": self._trade_get(t, "r_mult", 0.0),
                "highest_r": self._trade_get(t, "highest_r", 0.0),
                "size": self._trade_get(t, "size", self._trade_get(t, "size_usd", self._trade_get(t, "notional_usd", 0.0))),
                "timestamp": self._iso_ts(self._trade_get(t, "timestamp", self._trade_get(t, "opened_at", None))),
            })

        closed_rows = []
        closed_trades = self._state_get(state, "closed_trades", {})
        if isinstance(closed_trades, dict):
            iterable = closed_trades.items()
        elif isinstance(closed_trades, list):
            iterable = enumerate(closed_trades)
        else:
            iterable = []
        for tid, t in iterable:
            row = dict(t) if isinstance(t, dict) else {"trade_id": tid, **t.__dict__}
            row.setdefault("trade_id", tid)
            closed_rows.append(row)

        payload = {"open": pd.DataFrame(open_rows), "closed": pd.DataFrame(closed_rows)}

        if self.callbacks["update_trades"]:
            self.callbacks["update_trades"](payload)
        self._publish_snapshot()

    # ------------------------------------------------------------------
    def _publish_snapshot(self):
        if self.telemetry is None:
            return
        try:
            self.telemetry.publish_snapshot(self.get_snapshot())
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Telemetry snapshot publish failed: %s", exc)
