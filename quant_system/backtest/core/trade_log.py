"""
TradeLog:
Structured container for all executed trades during backtests.
Supports open/close lifecycle, R-multiple computation, and DataFrame export.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from itertools import count
from quant_system.utils.logger import get_logger

LOG = get_logger("trade_log")


class TradeLog:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self._id_seq = count(1)
        LOG.info("TradeLog initialized.")

    # ---------------------------------------------------------------
    def append_open(
        self,
        pos,
        ts,
        tier: str,
        conf: float,
        evr: float,
        risk: float,
        leg: str = "core",
        regime: str = None,
        hazard: float = None,
        gates: dict = None,
        gate_reasons: list = None,
    ):
        rec = {
            "trade_id": pos.trade_id,
            "asset": pos.asset,
            "side": pos.side,
            "entry_ts": pd.to_datetime(ts),
            "exit_ts": pd.NaT,
            "entry_price": pos.entry_price,
            "exit_price": None,
            "pnl": None,
            "r": None,
            "tier": tier,
            "conf": conf,
            "evr": evr,
            "risk": risk,
            "override": None,
            "reason": None,
            "stop_price": pos.stop_price,
            "size_usd": pos.size_usd,
            "qty": pos.qty,
            "leg": leg,
            "regime": regime,
            "hazard_entry": hazard,
            "gates": gates or {},
            "gate_reasons": gate_reasons or [],
        }
        self.trades.append(rec)
        LOG.info(f"Open logged trade_id={pos.trade_id} side={pos.side} tier={tier}")

    # ---------------------------------------------------------------
    def append_close(
        self,
        pos,
        pnl: float,
        ts,
        exit_price: float,
        reason: str = "exit",
        override: Optional[str] = None,
        regime: Optional[str] = None,
    ):
        tid = pos.trade_id
        rec = next((t for t in self.trades if t["trade_id"] == tid), None)
        if not rec:
            LOG.warning(f"Trade {tid} not found in log on close.")
            return

        rec["exit_ts"] = pd.to_datetime(ts)
        rec["exit_price"] = exit_price
        rec["pnl"] = pnl
        rec["reason"] = reason
        rec["override"] = override
        rec["regime_exit"] = regime
        rec["r"] = self._compute_r(rec)
        LOG.info(f"Close logged trade_id={tid} pnl={pnl:.4f} r={rec['r']:.4f}")

    # ---------------------------------------------------------------
    def _compute_r(self, rec: Dict[str, Any]) -> float:
        risk = abs(rec["entry_price"] - rec["stop_price"])
        if not risk or risk <= 1e-9:
            return 0.0
        move = rec["exit_price"] - rec["entry_price"]
        if rec["side"] == "short":
            move = -move
        return float(move / risk)

    # ---------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
