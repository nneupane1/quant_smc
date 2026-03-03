"""
Compound/cooling policy:
 - compounds the active ticket through deployable equity
 - vaults profits back to the base ticket when danger is detected
 - blocks entries during cooling unless overrides are explicitly enabled
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional, Union

from quant_system.execution.risk.cooling_engine import CoolingEngine


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class CompoundCoolingPolicy:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        capital_cfg = exec_cfg.get("capital", {})
        policy_cfg = exec_cfg.get("compound_cooling", {})
        hazard_cfg = exec_cfg.get("hazard_trailing", {}).get("thresholds", {})
        tiers_cfg = exec_cfg.get("tiers", {})
        mpc_cfg = exec_cfg.get("mpc", {})

        self.enabled = bool(policy_cfg.get("enabled", True))
        self.base_ticket_usd = float(
            capital_cfg.get("base_ticket_usd", exec_cfg.get("starting_equity", 0.0))
        )
        self.min_profit_to_lock_usd = float(policy_cfg.get("min_profit_to_lock_usd", 1.0))
        self.require_signals = int(policy_cfg.get("require_signals", 2))
        self.edp_gap_min = float(policy_cfg.get("edp_gap_min", 0.0))
        self.hazard_trigger = float(
            policy_cfg.get("hazard_trigger", hazard_cfg.get("partial", hazard_cfg.get("tighten", 0.4)))
        )
        self.cvar_target = float(mpc_cfg.get("cvar_target", 0.0))
        self.drawdown_trigger = float(
            policy_cfg.get("drawdown_trigger", exec_cfg.get("cooling_dd_trigger", 0.0))
        )
        self.bar_minutes = int(policy_cfg.get("bar_minutes", 15))
        self.allow_moonshot_during_cooling = bool(
            policy_cfg.get("allow_moonshot_during_cooling", False)
        )
        self.allow_twoR_during_cooling = bool(
            policy_cfg.get("allow_twoR_during_cooling", False)
        )
        self.base_conf_threshold = float(
            tiers_cfg.get("B", {}).get("min_confluence", 0.0)
        )

        self.cooling = CoolingEngine(cfg)

    def _extract_metrics(self, row: Optional[Dict[str, Any]]) -> Dict[str, float]:
        row = row or {}
        hazard = float(row.get("hazard_score", row.get("hazard", 0.0)) or 0.0)
        edp = float(row.get("p_edp", row.get("prob_edp", 0.0)) or 0.0)
        eop = float(row.get("p_eop", row.get("prob_eop", 0.0)) or 0.0)
        cvar = float(row.get("cvar", 0.0) or 0.0)

        quantiles = row.get("quantiles")
        if cvar == 0.0 and isinstance(quantiles, dict):
            q05 = float(quantiles.get("q05", quantiles.get("q0.05", 0.0)) or 0.0)
            q10 = float(quantiles.get("q10", quantiles.get("q0.10", 0.0)) or 0.0)
            cvar = abs(min((q05 + q10) / 2.0, 0.0))
        else:
            cvar = abs(min(cvar, 0.0)) if cvar < 0 else cvar

        return {"hazard": hazard, "edp": edp, "eop": eop, "cvar": cvar}

    def additional_lock_needed(self, equity: float, locked_profit: float) -> float:
        total_target_lock = max(equity - self.base_ticket_usd, 0.0)
        return max(total_target_lock - locked_profit, 0.0)

    def evaluate_danger(
        self,
        *,
        dt,
        equity: float,
        free_capital: float,
        locked_profit: float,
        drawdown: float,
        row: Optional[Dict[str, Any]] = None,
        cooling_until=None,
    ) -> Dict[str, Any]:
        metrics = self._extract_metrics(row)
        result = {
            "triggered": False,
            "reason": None,
            "votes": [],
            "metrics": metrics,
            "profit": 0.0,
            "lock_amount": 0.0,
            "cooldown_bars": 0,
            "cooling_until": cooling_until,
        }

        if not self.enabled:
            return result

        profit = max(equity - self.base_ticket_usd, 0.0)
        result["profit"] = profit
        if profit < self.min_profit_to_lock_usd:
            return result

        drawdown_triggered = drawdown >= abs(self.drawdown_trigger)
        if drawdown_triggered:
            result["triggered"] = True
            result["reason"] = "drawdown"
        else:
            votes = []
            if metrics["hazard"] >= self.hazard_trigger:
                votes.append("hazard")
            if (metrics["edp"] - metrics["eop"]) >= self.edp_gap_min:
                votes.append("edp_over_eop")
            if self.cvar_target > 0 and metrics["cvar"] > self.cvar_target:
                votes.append("cvar")
            result["votes"] = votes
            if len(votes) >= self.require_signals:
                result["triggered"] = True
                result["reason"] = "+".join(votes)

        if not result["triggered"]:
            return result

        result["lock_amount"] = min(
            free_capital,
            self.additional_lock_needed(equity, locked_profit),
        )

        locked_reference = max(equity - self.base_ticket_usd, 0.0)
        cooldown_bars = self.cooling.compute_cooldown(
            current_equity=equity,
            base_equity=self.base_ticket_usd,
            locked_amount=locked_reference,
        )
        result["cooldown_bars"] = cooldown_bars
        next_cooling_until = dt + timedelta(minutes=cooldown_bars * self.bar_minutes)
        if result["cooling_until"] is None or next_cooling_until > result["cooling_until"]:
            result["cooling_until"] = next_cooling_until

        return result

    def allow_entry(self, *, now, cooling_until, conf: float, evr: Dict[str, Any], hazard: float) -> Dict[str, Any]:
        if cooling_until is None or now >= cooling_until:
            return {"allow": True, "override": None, "remaining_bars": 0}

        if not self.allow_moonshot_during_cooling and not self.allow_twoR_during_cooling:
            remaining_seconds = max((cooling_until - now).total_seconds(), 0.0)
            remaining_bars = max(int(remaining_seconds // (self.bar_minutes * 60)), 1)
            return {"allow": False, "override": None, "remaining_bars": remaining_bars}

        remaining_seconds = max((cooling_until - now).total_seconds(), 0.0)
        remaining_bars = max(int(remaining_seconds // (self.bar_minutes * 60)), 1)
        gate = self.cooling.cooling_gate(
            in_cooldown=True,
            cooldown_remaining=remaining_bars,
            conf=float(conf or 0.0),
            base_conf_thr=self.base_conf_threshold,
            evr=float(evr.get("evr", 0.0) or 0.0),
            medianR=float(evr.get("median_r", evr.get("median_R", 0.0)) or 0.0),
            hazard=float(hazard or 0.0),
        )
        gate["remaining_bars"] = remaining_bars

        if gate["override"] == "moonshot" and not self.allow_moonshot_during_cooling:
            return {"allow": False, "override": None, "remaining_bars": remaining_bars}
        if gate["override"] == "2R" and not self.allow_twoR_during_cooling:
            return {"allow": False, "override": None, "remaining_bars": remaining_bars}
        return gate
