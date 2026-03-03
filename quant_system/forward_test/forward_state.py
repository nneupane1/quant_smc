"""Canonical forward-test state containers and snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ForwardTradeState:
    trade_id: str
    asset: str
    side: str
    entry_price: float
    qty: float
    size_usd: float
    stop_price: float
    opened_at: datetime
    leg: str = "core"
    r_mult: float = 0.0
    highest_r: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardState:
    starting_capital: float = 20_000.0
    equity: float = 20_000.0
    free_capital: float = 20_000.0
    locked_profit: float = 0.0
    max_drawdown: float = 0.0
    current_risk_mode: Any = None
    current_hedge_ratio: float = 0.0
    cooling_to: Optional[datetime] = None
    open_trades: Dict[str, ForwardTradeState] = field(default_factory=dict)
    closed_trades: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exposures: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "equity": self.equity,
            "free_capital": self.free_capital,
            "locked_profit": self.locked_profit,
            "max_drawdown": self.max_drawdown,
            "risk_mode": self.current_risk_mode,
            "hedge_ratio": self.current_hedge_ratio,
            "cooling_to": self.cooling_to,
            "open_trades": self.open_trades,
            "closed_trades": self.closed_trades,
            "exposures": self.exposures,
        }
