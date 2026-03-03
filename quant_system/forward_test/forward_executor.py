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
    notional_usd: float
    qty: float
    opened_at: datetime
    metadata: Dict[str, Any]
    stop_price: float
    leg: str = "core"  # core or runner
    r_mult: float = 0.0
    highest_r: float = 0.0

    @property
    def size_usd(self) -> float:
        return self.notional_usd


class ForwardExecutor:
    _id = itertools.count(1)

    def __init__(self, config_loader=None):
        self.cfg = config_loader

    def open_position(
        self,
        asset: str,
        entry_price: float,
        notional_usd: float,
        row: Dict[str, Any],
        stop_price: float = None,
        leg: str = "core",
    ) -> ForwardPosition:
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
            notional_usd=notional_usd,
            qty=qty,
            opened_at=row.get("dt", datetime.utcnow()),
            metadata={
                "evr": row.get("evr"),
                "conf": row.get("conf_score"),
                "median_r": row.get("median_r"),
                "p_bos_cont": row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)),
                "initial_stop": stop_price,
                "asset": asset,
                "side": side,
                "entry_price": entry_price,
                "size_usd": notional_usd,
                "leg": leg,
            },
            stop_price=stop_price,
            leg=leg
        )
        LOG.info(f"[ForwardExecutor] Opened {trade_id} {side} {asset} qty={qty:.4f} px={entry_price:.2f}")
        return pos

    def exit_position(self, pos: ForwardPosition, price: float) -> Dict[str, float]:
        """
        Close and return value + pnl (used to refill free_capital).
        """
        return self.exit_position_at(pos, price, datetime.utcnow())

    def exit_position_at(self, pos: ForwardPosition, price: float, exit_ts: datetime) -> Dict[str, float]:
        """
        Close at a deterministic timestamp for replay/forward parity.
        """
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - price) * pos.qty
        value = pos.notional_usd + pnl
        risk = abs(pos.entry_price - (pos.metadata.get("initial_stop") or pos.stop_price or pos.entry_price))
        r_mult = (pnl / (risk * pos.qty)) if risk > 0 and pos.qty > 0 else 0.0
        LOG.info(f"[ForwardExecutor] Exited {pos.trade_id} at {price:.2f}, value={value:.2f}, pnl={pnl:.2f}")
        return {
            "trade_id": pos.trade_id,
            "asset": pos.asset,
            "side": pos.side,
            "leg": pos.leg,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "entry_ts": pos.opened_at,
            "exit_ts": exit_ts,
            "qty": pos.qty,
            "size_usd": pos.notional_usd,
            "value": value,
            "pnl": pnl,
            "r": r_mult,
            "r_mult": r_mult,
            "stop_price": pos.stop_price,
            "conf": pos.metadata.get("conf"),
            "evr": pos.metadata.get("evr"),
            "tier": pos.metadata.get("tier"),
            "reason": pos.metadata.get("reason"),
        }

    def mark_to_market(self, pos: ForwardPosition, price: float) -> float:
        """
        Current market value of the position.
        """
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - price) * pos.qty
        return pos.notional_usd + pnl
