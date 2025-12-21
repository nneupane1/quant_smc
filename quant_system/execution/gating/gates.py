"""
GateEvaluator
--------------
Applies the 12h→6h→1h gating ladder before 15m execution.
Defaults are gentle; if a signal is missing we allow but record the reason.
"""

from typing import Any, Dict, Union
import pandas as pd


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class GateEvaluator:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        self.gates = cfg.get("execution", {}).get("gates", {})

        # Sensible defaults if not provided
        self.ocean_cfg = self.gates.get(
            "ocean_12h",
            {"trend_prob_min": 0.45, "tox_max": 0.35},
        )
        self.waves_cfg = self.gates.get(
            "waves_6h",
            {"zone_score_min": 0.60},
        )
        self.flow_cfg = self.gates.get(
            "flow_1h",
            {"body_min": 0.60, "vol_z_min": 0.80, "freshness_bars": 4},
        )

    @staticmethod
    def _get(row: pd.Series, keys, default=None):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return default

    def _ocean_ok(self, row: pd.Series, side: str, reasons: list) -> bool:
        trend_up = self._get(row, ["p_trend_up_12h", "p_trend_up"], 0.0)
        trend_down = self._get(row, ["p_trend_down_12h", "p_trend_down"], 0.0)
        tox = self._get(row, ["toxicity_12h", "tox_12h"], 0.0)

        if side == "long":
            if trend_up < self.ocean_cfg["trend_prob_min"]:
                reasons.append("ocean_trend_low")
                return False
        else:
            if trend_down < self.ocean_cfg["trend_prob_min"]:
                reasons.append("ocean_trend_low")
                return False

        if tox is not None and tox > self.ocean_cfg["tox_max"]:
            reasons.append("ocean_tox_high")
            return False

        return True

    def _waves_ok(self, row: pd.Series, side: str, reasons: list) -> bool:
        bias = self._get(row, ["structure_bias_6h", "bias_6h", "structure_bias"])
        zone_score = self._get(row, ["zone_score_6h", "zone_score"], None)

        # Allow if we lack the feature
        if bias is None and zone_score is None:
            return True

        if bias:
            if side == "long" and str(bias).lower() not in {"up", "bull", "bullish"}:
                reasons.append("waves_bias_mismatch")
                return False
            if side == "short" and str(bias).lower() not in {"down", "bear", "bearish"}:
                reasons.append("waves_bias_mismatch")
                return False

        if zone_score is not None and zone_score < self.waves_cfg["zone_score_min"]:
            reasons.append("waves_zone_low")
            return False

        return True

    def _flow_ok(self, row: pd.Series, reasons: list) -> bool:
        flow_flag = self._get(row, ["flow_ok_1h", "flow_ok"], None)
        if flow_flag is not None:
            return bool(flow_flag)

        body = self._get(row, ["displacement_body_pct_1h", "flow_body_pct_1h"], None)
        vol_z = self._get(row, ["volume_z_1h", "flow_vol_z_1h"], None)
        age = self._get(row, ["flow_age_bars_1h", "flow_age_bars"], None)

        # If we can't compute, allow but note
        if body is None or vol_z is None or age is None:
            return True

        if body < self.flow_cfg["body_min"]:
            reasons.append("flow_body_low")
            return False
        if vol_z < self.flow_cfg["vol_z_min"]:
            reasons.append("flow_vol_low")
            return False
        if age > self.flow_cfg["freshness_bars"]:
            reasons.append("flow_stale")
            return False

        return True

    def evaluate(self, row_like: Union[pd.Series, Dict[str, Any]], side: str) -> Dict[str, Any]:
        row = row_like if isinstance(row_like, pd.Series) else pd.Series(row_like)
        reasons = []

        ocean = self._ocean_ok(row, side, reasons)
        waves = self._waves_ok(row, side, reasons)
        flow = self._flow_ok(row, reasons)

        passed = ocean and waves and flow
        return {
            "passed": passed,
            "reasons": reasons,
            "checks": {"ocean_12h": ocean, "waves_6h": waves, "flow_1h": flow},
        }
