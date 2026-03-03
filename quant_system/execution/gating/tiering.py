"""
TieringEngine
-------------
Applies A+/A/B tiering using execution.tiers config and optional trade-rate
controls (cooldown, top_k_per_hour). Compatible with legacy decide() calls.
"""

import pandas as pd
from typing import Dict, Any, Union
import math

from quant_system.utils.logger import get_logger

LOG = get_logger("tiering")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class TieringEngine:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})

        # Preferred (new) shape
        self.tiers = exec_cfg.get("tiers", {})
        trade_rate = exec_cfg.get("trade_rate", {})

        # Legacy shape support
        if not self.tiers and "tiering" in exec_cfg:
            tcfg = exec_cfg["tiering"]
            self.tiers = tcfg.get("tier_rules", {})
            trade_rate = {"cooldown_bars": tcfg.get("cooldown_bars", 0), "top_k_per_hour": tcfg.get("top_k_per_hour", 1_000)}

        self.cooldown_bars = int(trade_rate.get("cooldown_bars", 0))
        self.top_k_per_hour = int(trade_rate.get("top_k_per_hour", 10_000))

        self.last_trade_bar = None
        self.hourly_count = {}

    # ------------------------------------------------------------ #
    def _cooldown_ok(self, bar_index: int) -> bool:
        if self.cooldown_bars <= 0:
            return True
        if self.last_trade_bar is None:
            return True
        return (bar_index - self.last_trade_bar) >= self.cooldown_bars

    def _topk_ok(self, ts: pd.Timestamp) -> bool:
        if self.top_k_per_hour <= 0:
            return True
        hour_key = ts.replace(minute=0, second=0, microsecond=0)
        cnt = self.hourly_count.get(hour_key, 0)
        return cnt < self.top_k_per_hour

    def _record(self, ts: pd.Timestamp, bar_index: int):
        self.last_trade_bar = bar_index
        hour_key = ts.replace(minute=0, second=0, microsecond=0)
        self.hourly_count[hour_key] = self.hourly_count.get(hour_key, 0) + 1

    # ------------------------------------------------------------ #
    def classify(
        self,
        row: pd.Series,
        confluence_pass: bool,
        evr_result: Dict[str, Any],
        hazard_score: float,
        bar_index: int,
    ) -> Dict[str, Any]:
        ts = row.get("dt") or row.name
        evr = evr_result.get("evr")
        med_r = evr_result.get("median_r")
        conf = row.get("confluence_score", row.get("conf_score", 0.0))

        conf = 0.0 if conf is None or (isinstance(conf, float) and math.isnan(conf)) else float(conf)
        evr = float("-inf") if evr is None or (isinstance(evr, float) and math.isnan(evr)) else float(evr)
        med_r = float("-inf") if med_r is None or (isinstance(med_r, float) and math.isnan(med_r)) else float(med_r)
        hazard_score = 1.0 if hazard_score is None or (isinstance(hazard_score, float) and math.isnan(hazard_score)) else float(hazard_score)

        if ts is None:
            ts = pd.Timestamp.utcnow()

        if not confluence_pass:
            return {"tier": "B", "execute": False, "reason": "confluence_fail"}

        if not self._cooldown_ok(bar_index):
            return {"tier": "B", "execute": False, "reason": "cooldown"}

        if not self._topk_ok(ts):
            return {"tier": "B", "execute": False, "reason": "topk_limit"}

        aplus = self.tiers.get("Aplus", {})
        a = self.tiers.get("A", {})
        b = self.tiers.get("B", {})

        def _pass(thr):
            return (
                conf >= float(thr.get("min_confluence", 0.0))
                and evr >= float(thr.get("min_evr", 0.0))
                and med_r >= float(thr.get("min_medianR", 0.0))
                and hazard_score <= float(thr.get("max_hazard", 1.0))
            )

        if _pass(aplus):
            self._record(ts, bar_index)
            return {"tier": "A+", "execute": bool(aplus.get("auto_execute", True)), "reason": "Aplus"}

        if _pass(a):
            self._record(ts, bar_index)
            return {"tier": "A", "execute": bool(a.get("auto_execute", False)), "reason": "A"}

        if _pass(b):
            return {"tier": "B", "execute": bool(b.get("auto_execute", False)), "reason": "B"}

        return {"tier": "skip", "execute": False, "reason": "below_thresholds"}

    # Compatibility with earlier interface
    def decide(self, *args, **kwargs):
        """
        Legacy helper:
            decide(conf_score, evr_dict) -> returns tier string
        """
        if len(args) == 2 and not kwargs:
            conf_score, evr = args
            dummy_row = pd.Series({"confluence_score": conf_score, "dt": pd.Timestamp.utcnow()})
            result = self.classify(
                row=dummy_row,
                confluence_pass=True,
                evr_result=evr,
                hazard_score=0.0,
                bar_index=0,
            )
            return result.get("tier", "skip")
        return self.classify(*args, **kwargs)

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "last_trade_bar": self.last_trade_bar,
            "hourly_count": {str(k): v for k, v in self.hourly_count.items()},
        }
