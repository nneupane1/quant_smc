"""
PositionSizer
-------------
R-based position sizing with toxicity/stretch adjustments. Defaults are
applied when config.execution.position_sizer is absent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from quant_system.utils.logger import get_logger

LOG = get_logger("position_sizer")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class PositionSizer:
    def __init__(self, config: Union[Dict[str, Any], Any] = None):
        cfg = _as_dict(config) if config is not None else {}
        exec_cfg = cfg.get("execution", {})
        scfg = exec_cfg.get("position_sizer", {})

        self.tox_scale = float(scfg.get("toxicity_scale", 0.0))
        self.stretch_scale = float(scfg.get("ema_stretch_scale", 0.0))
        self.min_qty = float(scfg.get("min_qty", 0.0))
        self.max_leverage = float(scfg.get("max_leverage", 5.0))
        self.session_sizing = exec_cfg.get("session_policy", {}).get("sizing", {})

    @staticmethod
    def _session_bucket(row: pd.Series) -> str:
        raw = str(row.get("session_bucket", "") or "").strip().lower()
        if raw in {"dead_zone", "pre_expansion", "expansion", "overlap"}:
            return raw
        if bool(row.get("session_overlap", 0)):
            return "overlap"
        if bool(row.get("session_pre_expansion", 0)):
            return "pre_expansion"
        if bool(row.get("session_expansion", 0)) or bool(row.get("session_london", 0)) or bool(row.get("session_ny", 0)):
            return "expansion"
        return "dead_zone"

    def _session_multiplier(self, row: pd.Series) -> float:
        quality = row.get("session_quality_multiplier")
        if quality is not None:
            try:
                q = float(quality)
                if np.isfinite(q) and q > 0:
                    return q
            except Exception:
                pass
        bucket = self._session_bucket(row)
        if not isinstance(self.session_sizing, dict):
            return 1.0
        bucket_cfg = self.session_sizing.get(bucket, {})
        if not isinstance(bucket_cfg, dict):
            return 1.0
        return float(bucket_cfg.get("qty_multiplier", 1.0))

    def _adjust_for_regime(self, qty: float, row: pd.Series) -> float:
        tox = float(row.get("toxicity_12h", 0.0))
        stretch = abs(float(row.get("ema_dist_z", 0.0)))
        qty *= (1.0 - self.tox_scale * tox)
        qty *= (1.0 - self.stretch_scale * stretch)
        qty *= max(self._session_multiplier(row), 0.0)
        return max(qty, 0.0)

    def _compute_qty(self, equity: float, risk_mode: float, stop_dist: float, px: float) -> float:
        if not np.isfinite(equity) or equity <= 0:
            return 0.0
        if not np.isfinite(risk_mode):
            risk_mode = 0.0
        if not np.isfinite(stop_dist) or stop_dist <= 0:
            return 0.0
        if not np.isfinite(px) or px <= 0:
            return 0.0
        position_value = equity * risk_mode
        raw_qty = position_value / stop_dist

        # Leverage bound
        lev = (raw_qty * px) / equity if equity else 0.0
        if lev > self.max_leverage:
            raw_qty = (self.max_leverage * equity) / px

        return max(raw_qty, self.min_qty)

    def size_position(
        self,
        row: pd.Series,
        equity: float,
        side: str,
        stop_price: float,
        risk_mode: float,
        hedge_ratio: float,
    ) -> Dict[str, Any]:
        px = float(row["close"])
        stop_price = float(stop_price)
        stop_dist = abs(px - stop_price)
        if (not np.isfinite(px)) or (not np.isfinite(stop_price)) or (not np.isfinite(stop_dist)) or stop_dist <= 0:
            return {"qty": 0.0, "hedge_qty": 0.0, "value": 0.0, "risk_dollars": 0.0, "leverage_estimate": 0.0}

        qty = self._compute_qty(equity, risk_mode, stop_dist, px)
        qty = self._adjust_for_regime(qty, row)
        hedge_qty = max(qty * hedge_ratio, 0.0)
        value = qty * px

        return {
            "qty": qty,
            "hedge_qty": hedge_qty,
            "value": value,
            "risk_dollars": qty * stop_dist,
            "leverage_estimate": (value / equity) if equity > 0 else 0.0,
        }

    # Compatibility shim
    def size(self, equity: float, risk_mode: float) -> float:
        return float(equity * risk_mode)
