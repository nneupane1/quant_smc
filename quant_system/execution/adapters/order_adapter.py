"""
OrderAdapter:
    Unified execution interface for:
        - Spot long trades
        - Directional shorts (spot-margin)
        - Perp hedge shorts

Handles:
    - order simulation (backtest / forward)
    - fees & slippage
    - funding / borrow APR (for directional shorts)
    - execution metadata for dashboards & logs
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from quant_system.utils.logger import log


class OrderAdapter:
    """
    Simulates execution fills and exposes a unified interface for trading.
    """

    def __init__(self, config: Dict[str, Any]):
        ocfg = config["execution"]["order_adapter"]

        # Fees (bps)
        self.taker_fee_bps = float(ocfg["taker_fee_bps"])
        self.maker_fee_bps = float(ocfg["maker_fee_bps"])

        # Slippage (bps)
        self.slippage_bps = float(ocfg["slippage_bps"])

        # Borrow APR for directional shorts
        self.borrow_apr = float(ocfg["borrow_apr"])  # annualized

        # Funding rate for perp hedge
        self.funding_rate = float(ocfg["funding_rate"])  # 8h funding, converted in calcs

        log("OrderAdapter initialized.")

    # ------------------------------------------------------------
    # Fee and slippage helpers
    # ------------------------------------------------------------
    def _calc_fee(self, notional: float, taker: bool = True) -> float:
        fee_bps = self.taker_fee_bps if taker else self.maker_fee_bps
        return (fee_bps / 10000.0) * notional

    def _calc_slippage(self, notional: float) -> float:
        return (self.slippage_bps / 10000.0) * notional

    # ------------------------------------------------------------
    # Borrow cost for directional shorts
    # ------------------------------------------------------------
    def _borrow_cost(self, notional: float, bars_held: int, bar_minutes: int = 15) -> float:
        """
        borrow_cost = notional * (APR / 365) * days_held
        """
        days = (bars_held * bar_minutes) / (60.0 * 24.0)
        return notional * (self.borrow_apr / 365.0) * days

    # ------------------------------------------------------------
    # Funding cost for perp hedge
    # ------------------------------------------------------------
    def _funding_cost(self, notional: float, bars_held: int, bar_minutes: int = 15) -> float:
        """
        Funding every 8h → approximate prorated funding.
        """
        days = (bars_held * bar_minutes) / (60.0 * 24.0)
        annualized = notional * self.funding_rate
        return annualized * days

    # ------------------------------------------------------------
    # Core fill simulation
    # ------------------------------------------------------------
    def _simulate_fill(self, px: float, qty: float, taker: bool = True) -> Dict[str, float]:
        notional = px * qty
        fee = self._calc_fee(notional, taker)
        slp = self._calc_slippage(notional)
        fill_px = px + (slp / qty if qty > 0 else 0)

        return {
            "fill_price": fill_px,
            "fee": fee,
            "slippage": slp,
            "notional": notional,
        }

    # ------------------------------------------------------------
    # Spot LONG execution
    # ------------------------------------------------------------
    def execute_long(self, px: float, qty: float) -> Dict[str, Any]:
        meta = self._simulate_fill(px, qty, taker=True)
        log(f"OrderAdapter: LONG qty={qty:.6f} at px={meta['fill_price']:.2f}")
        return {
            "side": "long",
            "qty": qty,
            "entry_price": meta["fill_price"],
            "fee": meta["fee"],
            "slippage": meta["slippage"],
            "notional": meta["notional"],
        }

    # ------------------------------------------------------------
    # Directional SHORT (spot-margin)
    # ------------------------------------------------------------
    def execute_short(self, px: float, qty: float) -> Dict[str, Any]:
        meta = self._simulate_fill(px, qty, taker=True)
        log(f"OrderAdapter: SHORT qty={qty:.6f} at px={meta['fill_price']:.2f}")
        return {
            "side": "short",
            "qty": qty,
            "entry_price": meta["fill_price"],
            "fee": meta["fee"],
            "slippage": meta["slippage"],
            "notional": meta["notional"],
            "borrow_cost": 0.0,   # computed on exit
        }

    # ------------------------------------------------------------
    # Hedge short (perp)
    # ------------------------------------------------------------
    def execute_hedge_short(self, px: float, qty: float) -> Dict[str, Any]:
        meta = self._simulate_fill(px, qty, taker=False)
        log(f"OrderAdapter: HEDGE SHORT qty={qty:.6f} at px={meta['fill_price']:.2f}")
        return {
            "side": "hedge_short",
            "qty": qty,
            "entry_price": meta["fill_price"],
            "fee": meta["fee"],
            "slippage": meta["slippage"],
            "notional": meta["notional"],
            "funding_cost": 0.0,  # computed on exit
        }

    # ------------------------------------------------------------
    # Exit logic (common for long / short / hedge)
    # ------------------------------------------------------------
    def exit_position(
        self,
        px: float,
        qty: float,
        side: str,
        entry_px: float,
        bars_held: int
    ) -> Dict[str, Any]:
        meta = self._simulate_fill(px, qty, taker=True)
        pnl = 0.0

        if side == "long":
            pnl = (meta["fill_price"] - entry_px) * qty

        elif side == "short":
            pnl = (entry_px - meta["fill_price"]) * qty
            borrow = self._borrow_cost(entry_px * qty, bars_held)
            pnl -= borrow
            meta["borrow_cost"] = borrow

        elif side == "hedge_short":
            pnl = (entry_px - meta["fill_price"]) * qty
            funding = self._funding_cost(entry_px * qty, bars_held)
            pnl -= funding
            meta["funding_cost"] = funding

        meta.update({
            "exit_price": meta["fill_price"],
            "pnl": pnl,
            "bars_held": bars_held,
        })

        log(
            f"OrderAdapter EXIT | side={side}, qty={qty:.6f}, pnl={pnl:.2f}, "
            f"entry={entry_px:.2f}, exit={meta['fill_price']:.2f}"
        )

        return meta
