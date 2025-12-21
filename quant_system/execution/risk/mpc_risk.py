"""
MPCRiskManager
---------------
CVaR-aware risk selector. Returns lock fraction, risk mode (R%), and hedge ratio.
Config compatible with execution.mpc (new) and execution.mpc_risk (legacy).
"""

import numpy as np
from typing import Dict, Any, Union

from quant_system.utils.logger import get_logger

LOG = get_logger("mpc_risk")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class MPCRiskManager:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        rcfg = exec_cfg.get("mpc") or exec_cfg.get("mpc_risk") or {}

        self.risk_modes = rcfg.get("risk_modes", {"low": 0.005, "medium": 0.01, "high": 0.015})
        self.lock_bounds = rcfg.get("lock_fraction_bounds", [0.0, 0.5])
        self.hedge_bounds = rcfg.get("hedge_ratio_bounds", [0.0, 0.8])
        self.cvar_target = float(rcfg.get("cvar_target", rcfg.get("cvar_cap", 0.02)))

        LOG.info("[MPC] risk_modes=%s lock_bounds=%s hedge_bounds=%s", self.risk_modes, self.lock_bounds, self.hedge_bounds)

    def _extract_quantiles(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Pull quantile forecasts from row if present. Accepts:
          - flat keys: q05, q10, q50, q90, q95
          - nested dict under 'quantiles'
        """
        if not isinstance(row, dict):
            return {}
        if "quantiles" in row and isinstance(row["quantiles"], dict):
            return {k: float(v) for k, v in row["quantiles"].items() if isinstance(v, (int, float))}
        keys = ["q05", "q10", "q50", "q90", "q95"]
        out = {}
        for k in keys:
            if k in row:
                try:
                    out[k] = float(row[k])
                except Exception:
                    pass
        return out

    def _select_risk_mode(self, hazard: float, edp: float, eop: float) -> float:
        # Simple heuristic: higher hazard/edp -> lower risk
        if hazard >= 0.6 or edp > eop:
            return float(self.risk_modes.get("low", 0.005))
        if hazard >= 0.4:
            return float(self.risk_modes.get("medium", 0.01))
        return float(self.risk_modes.get("high", 0.015))

    def _select_lock(self, edp: float) -> float:
        lo, hi = self.lock_bounds
        return float(np.clip(edp, lo, hi))

    def _select_hedge(self, hazard: float) -> float:
        lo, hi = self.hedge_bounds
        # scale hedge with hazard
        return float(np.clip(hazard, lo, hi))

    def _compute_cvar(self, quantiles: Dict[str, float]) -> float:
        """
        CVaR_0.95 approximated using q05 and q10 tail.
        CVaR ≈ mean(q05, q10).
        """
        if not quantiles:
            return 0.0
        q05 = quantiles.get("q05", quantiles.get("q0.05", 0.0))
        q10 = quantiles.get("q10", quantiles.get("q0.10", 0.0))
        return float((q05 + q10) / 2)

    def decide(
        self,
        equity: float,
        free_capital: float,
        locked_profit: float,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        hazard = float(row.get("hazard_score", row.get("hazard", 0.0)))
        edp = float(row.get("p_edp", row.get("prob_edp", 0.0)))
        eop = float(row.get("p_eop", row.get("prob_eop", 0.0)))
        quantiles = self._extract_quantiles(row)
        cvar = self._compute_cvar(quantiles)

        risk = self._select_risk_mode(hazard, edp, eop)
        lock = self._select_lock(edp)
        hedge = self._select_hedge(hazard)

        # If CVaR breaches target, force lower risk & higher hedge
        if cvar > self.cvar_target:
            risk = min(risk, float(self.risk_modes.get("low", risk)))
            hedge = max(hedge, self.hedge_bounds[1])
            lock = max(lock, self.lock_bounds[0])

        return {
            "lock_pct": lock,
            "lock_fraction": lock,
            "risk_mode": risk,
            "hedge_ratio": hedge,
            "reason": f"hazard={hazard:.3f}, edp={edp:.3f}, eop={eop:.3f}, cvar={cvar:.4f}",
            "quantiles": quantiles,
            "cvar": cvar,
        }

    # Compatibility wrapper
    def compute(self, *args, **kwargs):
        # legacy signature compute(equity, free_capital, locked_profit, row)
        if len(args) >= 4:
            return self.decide(args[0], args[1], args[2], args[3])
        return self.decide(
            kwargs.get("equity", 0.0),
            kwargs.get("free_capital", 0.0),
            kwargs.get("locked_profit", 0.0),
            kwargs.get("row", {}),
        )
