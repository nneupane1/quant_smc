"""
forward_executor.py

Lightweight paper-trade executor used by ForwardEngine.
Keeps position accounting consistent with ForwardEngine's equity math:
 - free_capital is reduced by notional at entry
 - mark_to_market returns full current value (qty * price)
 - exit_position returns full current value to be added back to free_capital
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
import itertools

from quant_system.utils.logger import get_logger

LOG = get_logger("forward_executor")


@dataclass
class ForwardPosition:
    trade_id: str
    asset: str
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    opened_at: datetime
    metadata: Dict[str, Any]
    leg: str = "core"  # core or runner


class ForwardExecutor:
    _id = itertools.count(1)

    def __init__(self, config_loader=None):
        self.cfg = config_loader

    def open_position(self, asset: str, entry_price: float, notional_usd: float, row: Dict[str, Any], leg: str = "core") -> ForwardPosition:
        """
        Create a paper position. Qty is denominated in base asset.
        """
        trade_id = f"fwd_{next(self._id)}"
        side = row.get("side", "long")
        qty = notional_usd / max(entry_price, 1e-9)
        pos = ForwardPosition(
            trade_id=trade_id,
            asset=asset,
            side=side,
            entry_price=entry_price,
            qty=qty,
            opened_at=row.get("dt", datetime.utcnow()),
            metadata={"evr": row.get("evr"), "conf": row.get("conf_score")},
            leg=leg
        )
        LOG.info(f"[ForwardExecutor] Opened {trade_id} {side} {asset} qty={qty:.4f} px={entry_price:.2f}")
        return pos

    def exit_position(self, pos: ForwardPosition, price: float) -> Dict[str, float]:
        """
        Close and return value + pnl (used to refill free_capital).
        """
        value = pos.qty * price
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - price) * pos.qty
        LOG.info(f"[ForwardExecutor] Exited {pos.trade_id} at {price:.2f}, value={value:.2f}, pnl={pnl:.2f}")
        return {"value": value, "pnl": pnl}

    def mark_to_market(self, pos: ForwardPosition, price: float) -> float:
        """
        Current market value of the position.
        """
        return pos.qty * price
