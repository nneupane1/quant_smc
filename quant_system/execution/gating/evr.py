"""
EVRCalculator
--------------
Computes stop/targets/R-multiples and EVR using the current execution config.
Designed to be resilient to missing fields; will return NaNs instead of raising.
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("evr")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class EVRCalculator:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})

        self.stop_cfg = exec_cfg.get("stops_targets", {})
        self.evr_cfg = exec_cfg.get("evr", {})

        self.stop_mult = float(self.stop_cfg.get("stop_atr_mult", 0.8))
        bounds = self.stop_cfg.get("stop_atr_bounds", [0.4, 2.0])
        self.stop_bounds = (float(bounds[0]), float(bounds[1]))
        self.target_order = self.stop_cfg.get(
            "target_order", ["swing", "fvg", "liquidity", "range"]
        )

        self.include_fees = bool(self.evr_cfg.get("include_fees", True))
        self.include_slippage = bool(self.evr_cfg.get("include_slippage", True))
        self.include_borrow = bool(self.evr_cfg.get("include_borrow_costs", True))
        self.fee_bps = self.evr_cfg.get("fee_bps", {"taker": 4, "maker": 2})
        self.slip_bps = self.evr_cfg.get("slippage_bps", {"limit_hit": 1.5})
        self.borrow_apr_cap = float(self.evr_cfg.get("borrow_apr_cap", 0.0))

    # ------------------------------------------------------------------ #
    def _compute_stop(self, row: pd.Series, side: str) -> float:
        atr = float(row.get("atr", np.nan))
        if np.isnan(atr):
            return np.nan
        mult = np.clip(self.stop_mult, *self.stop_bounds)
        px = float(row["close"])
        if side == "long":
            return float(px - mult * atr)
        return float(px + mult * atr)

    def _targets(self, row: pd.Series, side: str) -> List[float]:
        mapping = {
            "swing": "swing_target",
            "fvg": "fvg_target",
            "liquidity": "liquidity_pool",
            "range": "range_edge",
        }
        out = []
        px = float(row["close"])
        for key in self.target_order:
            field = mapping.get(key, key)
            val = row.get(field)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val = float(val)
            if side == "long" and val > px:
                out.append(val)
            elif side == "short" and val < px:
                out.append(val)
        return out

    def _r_values(self, entry: float, stop: float, targets: List[float], side: str) -> List[float]:
        risk = abs(entry - stop)
        if risk <= 0:
            return []
        r_list = []
        for t in targets:
            if side == "long":
                r = (t - entry) / risk
            else:
                r = (entry - t) / risk
            r_list.append(float(r))
        return r_list

    def _cost_in_r(self, entry: float, stop: float, side: str) -> float:
        risk = abs(entry - stop)
        if risk <= 0:
            return 0.0

        fee = (self.fee_bps.get("taker", 0) / 10000.0) if self.include_fees else 0.0
        slip = (self.slip_bps.get("market", 0) / 10000.0) if self.include_slippage else 0.0
        borrow = (self.borrow_apr_cap / 365 / 24 / 4) if (self.include_borrow and side == "short") else 0.0

        total = (fee + slip + borrow) * entry
        return float(total / risk)

    # ------------------------------------------------------------------ #
    def compute_evr(self, row: pd.Series, side: str) -> Dict[str, Any]:
        px = float(row["close"])
        stop = self._compute_stop(row, side)
        targets = self._targets(row, side)
        r_vals = self._r_values(px, stop, targets, side)

        if len(r_vals) == 0:
            return {
                "stop_price": stop,
                "targets": targets,
                "r_values": [],
                "evr": np.nan,
                "median_r": np.nan,
            }

        p_cont = float(row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)))
        cost_r = self._cost_in_r(px, stop, side)

        evr = (p_cont * float(np.mean(r_vals))) - ((1 - p_cont) * 1.0) - cost_r
        median_r = float(np.median(r_vals))

        LOG.debug(
            "[EVR] side=%s evr=%.3f medianR=%.3f stop=%.3f targets=%s",
            side,
            evr,
            median_r,
            stop,
            targets,
        )
        return {
            "stop_price": stop,
            "targets": targets,
            "r_values": r_vals,
            "evr": evr,
            "median_r": median_r,
        }

    # Compatibility helpers
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        # legacy signature: compute(models, row)
        if args and isinstance(args[-1], dict):
            row = pd.Series(args[-1])
        else:
            row = kwargs.get("row", pd.Series({}))
        side = kwargs.get("side", row.get("side", "long"))
        return self.compute_evr(row, side)
