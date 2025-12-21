"""
ConfluenceEngine
----------------
Unified confluence scorer that is tolerant to both legacy and new config
shapes. It produces a numeric score and a boolean pass flag used by tiering.
"""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("confluence")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Accept ConfigLoader/ConfigManager/dict and return a plain dict."""
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class ConfluenceEngine:
    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        self.conf_cfg = exec_cfg.get("confluence", {})
        tiers_cfg = exec_cfg.get("tiers", {})

        self.weights = {
            "specialist": float(self.conf_cfg.get("weight_specialist", 1.0)),
            "meta": float(self.conf_cfg.get("weight_meta", 0.5)),
            "rsv": float(self.conf_cfg.get("weight_rsv", 0.5)),
            "smc": float(self.conf_cfg.get("weight_smc", 0.5)),
            "session": float(self.conf_cfg.get("weight_session", 1.0)),
        }

        self.session_weights = self.conf_cfg.get("session_weights", {})
        self.default_threshold = float(
            self.conf_cfg.get(
                "default_threshold",
                tiers_cfg.get("A", {}).get("min_confluence", 0.5),
            )
        )

        LOG.info(
            "[ConfluenceEngine] Loaded with weights=%s, default_threshold=%.3f",
            self.weights,
            self.default_threshold,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first(row: pd.Series, candidates):
        for key in candidates:
            if key in row and row[key] is not None:
                return float(row[key])
        return 0.0

    def _session_weight(self, session: str) -> float:
        return float(self.session_weights.get(session, 1.0)) * self.weights["session"]

    def _specialist_score(self, row: pd.Series) -> float:
        p_liq = self._first(row, ["p_liq_flow", "prob_liq_flow"])
        p_bos = self._first(row, ["p_bos_cont", "prob_bos_cont"])
        p_momo = self._first(row, ["p_momo", "prob_momo", "prob_micro_momentum"])
        probs = [p for p in [p_liq, p_bos, p_momo] if p is not None]
        return float(np.nanmean(probs)) if probs else 0.0

    def _meta_score(self, row: pd.Series) -> float:
        return self._first(row, ["prob_meta", "meta_prob"])

    def _regime_score(self, row: pd.Series) -> float:
        trend_up = self._first(row, ["p_trend_up"])
        expansion = self._first(row, ["p_expansion"])
        tox = self._first(row, ["toxicity_12h", "toxicity"])
        score = trend_up + 0.5 * expansion - 0.5 * tox
        return float(score)

    def _smc_score(self, row: pd.Series) -> float:
        struct = self._first(row, ["structure_density", "smc_strength", "smc_score"])
        zone_align = self._first(row, ["zone_alignment", "ob_alignment"])
        pd_region = self._first(row, ["pd_position"])
        return float(struct + 0.3 * zone_align + 0.2 * pd_region)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(self, row: pd.Series) -> Dict[str, Any]:
        """
        Returns:
            {
              "confluence_score": float,
              "passed": bool
            }
        """
        session_w = self._session_weight(row.get("session", "other"))

        score = (
            self.weights["specialist"] * self._specialist_score(row)
            + self.weights["meta"] * self._meta_score(row)
            + self.weights["rsv"] * self._regime_score(row)
            + self.weights["smc"] * self._smc_score(row)
        )

        score *= session_w
        passed = score >= self.default_threshold

        LOG.debug(
            "[ConfluenceEngine] score=%.4f passed=%s session_w=%.2f",
            score,
            passed,
            session_w,
        )
        return {"confluence_score": float(score), "passed": bool(passed)}

    # Compatibility for older code paths
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        row = args[-1] if args else kwargs.get("row")
        if not isinstance(row, pd.Series):
            row = pd.Series(row)
        out = self.evaluate(row)
        out["score"] = out["confluence_score"]
        out["conf_score"] = out["confluence_score"]
        out["allow"] = out["passed"]
        return out

    def compute_confluence(self, *args, **kwargs) -> Dict[str, Any]:
        return self.compute(*args, **kwargs)
