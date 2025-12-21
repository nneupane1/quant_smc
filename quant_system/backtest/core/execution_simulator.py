"""
ExecutionSimulator:
Deterministic intrabar simulator for historical backtests.
Handles market entry/exit with slippage + fees, stop checks, and mark-to-market.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np

from quant_system.utils.logger import get_logger

LOG = get_logger("execution_simulator")


@dataclass
class Position:
    trade_id: str
    asset: str
    side: str
    entry_price: float
    size_usd: float
    qty: float
    open_time: Any
    stop_price: float
    tier: Optional[str] = None
    conf: Optional[float] = None
    evr: Optional[float] = None
    risk: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionSimulator:
    def __init__(self, config: Dict[str, Any]):
        ec = config.get("execution", {}).get("simulator", {})
        self.maker_fee = float(ec.get("maker_fee", 0.0))
        self.taker_fee = float(ec.get("taker_fee", 0.0))
        self.slip_limit = float(ec.get("slippage_limit", 0.0))
        self.slip_market = float(ec.get("slippage_market", 0.0))
        self.path_mode = ec.get("path_mode", "nearest_extreme_first")
        self._id_seq = itertools.count(1)
        LOG.info("ExecutionSimulator initialized.")

    # ------------------------------------------------------------------
    def _market_fill(self, price: float, side: str) -> float:
        slip = price * self.slip_market
        fill = price + slip if side == "long" else price - slip
        fee = fill * self.taker_fee
        return fill + fee if side == "long" else fill - fee

    # ------------------------------------------------------------------
    def open_position(self, asset: str, side: str, entry_price: float, size_usd: float, row_ts, stop_price: float) -> Position:
        fill = self._market_fill(entry_price, side)
        qty = size_usd / fill if fill else 0.0
        trade_id = f"T{next(self._id_seq)}"
        LOG.info(f"Open position {trade_id} {side} {asset} @ {fill:.4f} qty={qty:.4f}")
        return Position(
            trade_id=trade_id,
            asset=asset,
            side=side,
            entry_price=fill,
            size_usd=size_usd,
            qty=qty,
            open_time=row_ts,
            stop_price=stop_price,
        )

    # ------------------------------------------------------------------
    def exit_position(self, pos: Position, exit_price: float, reason: str = "exit") -> Dict[str, Any]:
        fee = abs(exit_price) * self.taker_fee
        pnl = (exit_price - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - exit_price) * pos.qty
        pnl -= fee
        LOG.info(f"Exit {pos.trade_id} reason={reason} exit={exit_price:.4f} pnl={pnl:.4f}")
        return {"pnl": pnl, "exit_price": exit_price, "reason": reason}

    # ------------------------------------------------------------------
    def mark_to_market(self, pos: Position, current_price: float) -> float:
        """
        Returns current position equity (cost + unrealized pnl).
        """
        if current_price is None or np.isnan(current_price):
            return pos.size_usd
        unrealized = (current_price - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - current_price) * pos.qty
        return pos.size_usd + unrealized

    # ------------------------------------------------------------------
    def simulate_ohlc_path(self, o: float, h: float, l: float, c: float, prev_price: float):
        """
        Deterministic OHLC path based on nearest-extreme-first.
        Path: O -> nearest extreme (H or L) -> other extreme -> C.
        """
        dist_h = abs(prev_price - h)
        dist_l = abs(prev_price - l)
        return [o, h, l, c] if dist_h < dist_l else [o, l, h, c]

    # ------------------------------------------------------------------
    def check_stop(self, row, pos: Position) -> Optional[Dict[str, Any]]:
        """
        Checks whether the stop is hit intrabar using 1m OHLC path.
        Expects row to have open_1m/high_1m/low_1m/close_1m.
        """
        try:
            o = float(row["open_1m"])
            h = float(row["high_1m"])
            l = float(row["low_1m"])
            c = float(row["close_1m"])
        except Exception:
            return None

        path = self.simulate_ohlc_path(o, h, l, c, prev_price=pos.entry_price)
        stop = pos.stop_price

        if pos.side == "long":
            for p in path:
                if p <= stop:
                    exit_price = stop - (stop * self.slip_market)
                    return self.exit_position(pos, exit_price, reason="stop_hit")
        else:
            for p in path:
                if p >= stop:
                    exit_price = stop + (stop * self.slip_market)
                    return self.exit_position(pos, exit_price, reason="stop_hit")
        return None
