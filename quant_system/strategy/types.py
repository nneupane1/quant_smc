from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Leg:
    ts: str
    side: Side
    qty: float
    entry_price: float
    risk_price: float       # per-leg structural stop
    tokens_risked: float    # risk in currency/token units
    comment: str = ""


@dataclass
class Position:
    id: str
    side: Side
    legs: List[Leg] = field(default_factory=list)
    base_stop: Optional[float] = None
    avg_entry: Optional[float] = None
    total_qty: float = 0.0
    realized_pnl: float = 0.0
    partial_taken: bool = False
    last_add_r_multiple: float = 0.0

    def recalc(self):
        if not self.legs:
            self.avg_entry = None
            self.total_qty = 0.0
            return
        notional = sum(l.qty * l.entry_price for l in self.legs)
        qty = sum(l.qty for l in self.legs)
        self.total_qty = qty
        self.avg_entry = notional / qty if qty else None
