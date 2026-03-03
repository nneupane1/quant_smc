"""Unified execution-cost adapter for simulated long, short, and hedge legs."""

from typing import Any, Dict, Union

from quant_system.utils.logger import log


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class OrderAdapter:
    """
    Simulates execution fills and exposes a unified interface for trading.
    """

    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        ocfg = exec_cfg.get("order_adapter", {})
        legacy_fees = exec_cfg.get("legacy_fees", {})

        # Fees (bps)
        self.taker_fee_bps = float(ocfg.get("taker_fee_bps", legacy_fees.get("taker", 0.0) * 10_000))
        self.maker_fee_bps = float(ocfg.get("maker_fee_bps", legacy_fees.get("maker", 0.0) * 10_000))

        # Slippage (bps)
        self.slippage_bps = float(ocfg.get("slippage_bps", legacy_fees.get("slippage_bps", 0.0)))

        # Borrow APR for directional shorts
        self.borrow_apr = float(ocfg.get("borrow_apr", exec_cfg.get("shorting", {}).get("directional_short", {}).get("apr_cap", 0.0)))

        # Funding rate for perp hedge
        self.funding_rate = float(ocfg.get("funding_rate", 0.0))  # 8h funding, converted in calcs

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

    def execute(self, side: str, px: float, qty: float, hedge: bool = False) -> Dict[str, Any]:
        """
        Compatibility entrypoint for callers that select the leg at runtime.
        """
        if hedge:
            return self.execute_hedge_short(px, qty)
        if side == "short":
            return self.execute_short(px, qty)
        return self.execute_long(px, qty)

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

        pnl -= meta["fee"]

        meta.update({
            "exit_price": meta["fill_price"],
            "pnl": pnl,
            "bars_held": bars_held,
            "value": qty * meta["fill_price"],
        })

        log(
            f"OrderAdapter EXIT | side={side}, qty={qty:.6f}, pnl={pnl:.2f}, "
            f"entry={entry_px:.2f}, exit={meta['fill_price']:.2f}"
        )

        return meta
