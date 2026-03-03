"""
Capital allocation policy for backtest/forward/live engines.

Default behavior is ticket-based:
 - start from a base ticket size (20k by default)
 - compound the ticket with deployable equity if enabled
 - optionally delegate to MPC later
"""

from typing import Any, Dict, Union


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class CapitalAllocator:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        capital_cfg = exec_cfg.get("capital", {})

        self.starting_equity = float(exec_cfg.get("starting_equity", capital_cfg.get("base_ticket_usd", 0.0)))
        self.base_ticket_usd = float(capital_cfg.get("base_ticket_usd", self.starting_equity))
        self.compound_ticket = bool(capital_cfg.get("compound_ticket", True))
        self.use_mpc = bool(capital_cfg.get("use_mpc", False))
        self.min_free_capital_usd = float(capital_cfg.get("min_free_capital_usd", 0.0))

    def allocate(
        self,
        equity: float,
        free_capital: float,
        locked_profit: float,
        row: Dict[str, Any] = None,
        mpc_manager: Any = None,
    ) -> Dict[str, Any]:
        if free_capital <= 0:
            return {
                "ticket_usd": 0.0,
                "hedge_ratio": 0.0,
                "lock_fraction": 0.0,
                "risk_mode": None,
                "mode": "blocked",
                "deployable_capital": 0.0,
                "ticket_multiple": 0.0,
            }

        deployable = max(equity - locked_profit, 0.0)
        ticket_usd = self.base_ticket_usd
        ticket_multiple = 1.0

        if self.compound_ticket and self.starting_equity > 0:
            ticket_multiple = deployable / self.starting_equity
            ticket_usd *= ticket_multiple

        hedge_ratio = 0.0
        lock_fraction = 0.0
        risk_mode = None
        mode = "ticket"

        if self.use_mpc and mpc_manager is not None and row is not None:
            mpc_out = mpc_manager.decide(
                equity=equity,
                free_capital=free_capital,
                locked_profit=locked_profit,
                row=row,
            )
            hedge_ratio = float(mpc_out.get("hedge_ratio", 0.0))
            lock_fraction = float(mpc_out.get("lock_fraction", mpc_out.get("lock_pct", 0.0)))
            risk_mode = mpc_out.get("risk_mode")
            if risk_mode is not None:
                ticket_usd = deployable * float(risk_mode)
            mode = "mpc"

        ticket_usd = min(ticket_usd, free_capital)
        if ticket_usd < self.min_free_capital_usd:
            ticket_usd = 0.0

        return {
            "ticket_usd": max(ticket_usd, 0.0),
            "hedge_ratio": hedge_ratio,
            "lock_fraction": lock_fraction,
            "risk_mode": risk_mode,
            "mode": mode,
            "deployable_capital": deployable,
            "ticket_multiple": max(ticket_multiple, 0.0),
        }
