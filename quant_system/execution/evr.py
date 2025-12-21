"""
EVRCalculator:
    Computes Expected-R (EVR) and median-R for execution gating.
    Components:
        - stop placement (zone ± ATR*scale)
        - multi-TF target selection (15m → 1h → 6h → 12h)
        - structural liquidity/FVG targets
        - cost modeling in R space
        - EVR = Σ p_i * R_i  - (1 - p_cont)*1R  - cost_R
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from quant_system.data.kraken_client import log


class EVRCalculator:
    """
    Computes:
        - stop_loss_price
        - targets (in price terms)
        - R-values per target
        - EVR
        - median-R
    """

    def __init__(self, config: Dict[str, Any]):
        ecfg = config["execution"]["evr"]

        self.atr_mult_min = ecfg["atr_multiplier_bounds"]["min"]    # 0.4
        self.atr_mult_max = ecfg["atr_multiplier_bounds"]["max"]    # 2.0
        self.default_atr_mult = ecfg["default_atr_multiplier"]      # 0.8

        self.fee_bps = float(ecfg["fees_bps"])                      # taker/maker blended
        self.slippage_bps = float(ecfg["slippage_bps"])
        self.short_borrow_bps = float(ecfg["short_borrow_bps"])

        self.target_ranks = ecfg["target_priority"]

        log("EVRCalculator initialized.")

    # ------------------------------------------------------------
    # Stop computation
    # ------------------------------------------------------------
    def compute_stop(self, row: pd.Series, side: str) -> float:
        """
        Stop = far edge of the OB/FVG zone ± ATR * scale
        """
        atr = row.get("atr", np.nan)
        if np.isnan(atr):
            return np.nan

        # Zone fields filled upstream by structure_context
        zone_far = row.get("zone_far_price", np.nan)
        if np.isnan(zone_far):
            return np.nan

        atr_mult = np.clip(self.default_atr_mult, self.atr_mult_min, self.atr_mult_max)

        if side == "long":
            stop = zone_far - atr_mult * atr
        else:
            stop = zone_far + atr_mult * atr

        return float(stop)

    # ------------------------------------------------------------
    # Target selection hierarchy
    # ------------------------------------------------------------
    def _gather_targets(self, row: pd.Series, side: str) -> List[float]:
        """
        Target priority defined in YAML:
            - swing_1h
            - swing_6h
            - swing_12h
            - fvg_target
            - liquidity_pool
            - range_edge
        """

        targets = []
        for key in self.target_ranks:
            price = row.get(key, np.nan)
            if not np.isnan(price):
                targets.append(float(price))

        # Side-directional cleanup
        px = row["close"]
        if side == "long":
            targets = [t for t in targets if t > px]
        else:
            targets = [t for t in targets if t < px]

        return targets

    # ------------------------------------------------------------
    # Convert price targets → R-multiples
    # ------------------------------------------------------------
    def _compute_r_values(self, px_entry: float, stop: float, targets: List[float], side: str) -> List[float]:
        if np.isnan(px_entry) or np.isnan(stop):
            return []

        risk = abs(px_entry - stop)
        if risk <= 0:
            return []

        r_list = []
        for t in targets:
            if side == "long":
                r = (t - px_entry) / risk
            else:
                r = (px_entry - t) / risk
            r_list.append(float(r))

        return r_list

    # ------------------------------------------------------------
    # Convert execution costs to R-space
    # ------------------------------------------------------------
    def _cost_in_r(self, px_entry: float, stop: float, side: str) -> float:
        """
        Costs: fees + slippage + borrow (if short)
        Convert into R:
            cost_R = (cost_bps / 10000) * px_entry / risk
        """
        risk = abs(px_entry - stop)
        if risk <= 0:
            return 0.0

        base_bps = self.fee_bps + self.slippage_bps
        if side == "short":
            base_bps += self.short_borrow_bps

        cost_ratio = (base_bps / 10000.0) * px_entry
        return float(cost_ratio / risk)

    # ------------------------------------------------------------
    # EVR formula
    # ------------------------------------------------------------
    def compute_evr(self, row: pd.Series, side: str) -> Dict[str, Any]:
        """
        Returns dict:
            stop_price
            targets
            r_values
            evr
            median_r
        """

        px = row["close"]
        stop = self.compute_stop(row, side)
        targets = self._gather_targets(row, side)
        r_vals = self._compute_r_values(px, stop, targets, side)

        # No targets → no EVR
        if len(r_vals) == 0:
            return {
                "stop_price": stop,
                "targets": [],
                "r_values": [],
                "evr": np.nan,
                "median_r": np.nan,
            }

        # Probability of continuation (from BOS-cont model)
        p_cont = float(row.get("p_bos_cont", 0.0))

        cost_r = self._cost_in_r(px, stop, side)

        # Arithmetic EVR formula:
        # EVR = sum(p_i * R_i) - (1 - p_cont)*1R - cost_R
        evr = (p_cont * np.mean(r_vals)) - ((1 - p_cont) * 1.0) - cost_r
        median_r = float(np.median(r_vals))

        log(f"EVR: side={side}, EVR={evr:.3f}, medianR={median_r:.3f}, stop={stop}")

        return {
            "stop_price": stop,
            "targets": targets,
            "r_values": r_vals,
            "evr": evr,
            "median_r": median_r,
        }
        
        
        
        
        
        
"""
evr.py — Full reasoning version
"""

from typing import Dict, Any
import numpy as np


class EVREngine:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def compute_evr(self, conf, df, smc) -> Dict[str, Any]:

        targets = smc["context"]["targets"]
        probs = conf["breakdown"]["raw_inputs"]
        p_cont = max(0.0, min(1.0, probs["p_bos"]))  

        r_list = []
        p_list = []

        for t in targets:
            r_list.append(t["r_multiple"])
            p_list.append(t["prob"])

        if not r_list:
            evr = 0
            median_r = 0
        else:
            evr = sum([p_list[i] * r_list[i] for i in range(len(r_list))])
            median_r = np.median(r_list)

        reasoning = {
            "targets": targets,
            "p_cont": p_cont,
            "evr_components": {
                "r_list": r_list,
                "p_list": p_list
            }
        }

        moonshot_flag = median_r >= self.cfg["execution"]["moonshot_r"]

        return {
            "evr": evr,
            "median_r": median_r,
            "evr_min": self.cfg["execution"]["min_evr"],
            "moonshot_flag": moonshot_flag,
            "reasoning": reasoning
        }

