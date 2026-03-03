"""
Explicit profit-taking ladder for the main engines.

Uses the existing core/runner split:
 - move stops to breakeven after 2R
 - take the core leg at 2R or 3R depending on setup strength
 - leave the runner for a 6R-10R target while hazard trailing manages exits
"""

from typing import Any, Dict, Union

from quant_system.strategy.utils import r_multiple


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class ProfitLadderManager:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        lcfg = cfg.get("execution", {}).get("profit_ladder", {})

        self.enabled = bool(lcfg.get("enabled", True))
        self.breakeven_after_r = float(lcfg.get("breakeven_after_r", 2.0))
        self.breakeven_buffer_r = float(lcfg.get("breakeven_buffer_r", 0.0))
        self.core_take_r = float(lcfg.get("core_take_r", 2.0))
        self.strong_core_take_r = float(lcfg.get("strong_core_take_r", 3.0))
        self.runner_target_min_r = float(lcfg.get("runner_target_min_r", 6.0))
        self.runner_target_max_r = float(lcfg.get("runner_target_max_r", 10.0))
        self.strong_conf_min = float(lcfg.get("strong_conf_min", 0.75))
        self.strong_bos_cont_min = float(lcfg.get("strong_bos_cont_min", 0.60))
        self.strong_median_r_min = float(lcfg.get("strong_median_r_min", 6.0))

    def _risk_per_unit(self, pos) -> float:
        initial_stop = pos.metadata.get("initial_stop", getattr(pos, "stop_price", None))
        if initial_stop is None:
            return 0.0
        return abs(float(pos.entry_price) - float(initial_stop))

    def _is_strong_setup(self, pos, row: Dict[str, Any]) -> bool:
        conf = float(
            pos.metadata.get(
                "conf",
                getattr(pos, "conf", row.get("confluence_score", row.get("conf_score", 0.0))),
            )
            or 0.0
        )
        bos_cont = float(
            pos.metadata.get(
                "p_bos_cont",
                row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)),
            )
            or 0.0
        )
        median_r = float(
            pos.metadata.get(
                "median_r",
                row.get("median_r", row.get("median_R", 0.0)),
            )
            or 0.0
        )
        return (
            conf >= self.strong_conf_min
            and bos_cont >= self.strong_bos_cont_min
            and median_r >= self.strong_median_r_min
        )

    def _target_r(self, pos, row: Dict[str, Any]) -> float:
        leg = pos.metadata.get("leg", getattr(pos, "leg", "core"))
        strong = self._is_strong_setup(pos, row)
        if leg == "core":
            return self.strong_core_take_r if strong else self.core_take_r

        median_r = float(
            pos.metadata.get(
                "median_r",
                row.get("median_r", row.get("median_R", self.runner_target_min_r)),
            )
            or self.runner_target_min_r
        )
        return max(self.runner_target_min_r, min(self.runner_target_max_r, median_r))

    def _price_for_r(self, pos, r_target: float) -> float:
        risk = self._risk_per_unit(pos)
        if risk <= 0:
            return float(pos.entry_price)
        if pos.side == "long":
            return float(pos.entry_price) + (r_target * risk)
        return float(pos.entry_price) - (r_target * risk)

    def _breakeven_stop(self, pos) -> float:
        risk = self._risk_per_unit(pos)
        if pos.side == "long":
            return float(pos.entry_price) + (self.breakeven_buffer_r * risk)
        return float(pos.entry_price) - (self.breakeven_buffer_r * risk)

    def evaluate(self, pos, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"exit": False, "reason": None, "exit_price": None, "new_stop": None, "r_now": 0.0}

        if "initial_stop" not in pos.metadata and getattr(pos, "stop_price", None) is not None:
            pos.metadata["initial_stop"] = pos.stop_price

        risk = self._risk_per_unit(pos)
        if risk <= 0:
            return {"exit": False, "reason": None, "exit_price": None, "new_stop": None, "r_now": 0.0}

        current_price = float(row["close"])
        r_now = r_multiple(float(pos.entry_price), float(pos.metadata["initial_stop"]), current_price, pos.side)
        new_stop = None

        if not pos.metadata.get("breakeven_armed", False) and r_now >= self.breakeven_after_r:
            be_stop = self._breakeven_stop(pos)
            current_stop = float(getattr(pos, "stop_price", pos.metadata["initial_stop"]))
            if pos.side == "long":
                new_stop = max(current_stop, be_stop)
            else:
                new_stop = min(current_stop, be_stop)
            pos.metadata["breakeven_armed"] = True

        target_r = pos.metadata.get("take_profit_r")
        if target_r is None:
            target_r = self._target_r(pos, row)
            pos.metadata["take_profit_r"] = target_r

        if r_now >= float(target_r):
            return {
                "exit": True,
                "reason": f"{pos.metadata.get('leg', 'leg')}_tp_{float(target_r):.1f}R",
                "exit_price": self._price_for_r(pos, float(target_r)),
                "new_stop": new_stop,
                "r_now": r_now,
            }

        return {
            "exit": False,
            "reason": None,
            "exit_price": None,
            "new_stop": new_stop,
            "r_now": r_now,
        }
