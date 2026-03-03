"""
regime_rules.py
Maps HMM regime → model preference.

Config example:
rules:
  trend:
    long:
      prefer: ["model_trend_v3", "model_v2"]
    short:
      prefer: ["model_trend_v4"]
  range:
    prefer: ["model_range_v1"]
"""

from quant_system.utils.logger import get_logger

LOG = get_logger("regime_rules")


class RegimeRules:
    """Simple rule engine for selecting models based on regime."""

    def __init__(self, config: dict):
        self.config = config or {}
        self.min_prob = float(self.config.get("min_prob", 0.15))

    # --------------------------------------------------------------
    def apply(self, regime_probs: dict, vol_pctile: float, session: str, available_models=None):
        """
        Returns model_id or None.
        """

        regime_probs = self._normalize_regime_probs(regime_probs)
        if not regime_probs:
            return None
        available = set(available_models or [])

        # Highest-prob regime
        r = max(regime_probs, key=lambda k: regime_probs[k])
        if regime_probs.get(r, 0) < self.min_prob:
            return None

        if r not in self.config:
            return None

        cfg = self.config[r]

        # If direction matters (trend up/down), enrich conditions
        if r == "trend":
            if regime_probs.get("trend_up", 0) > regime_probs.get("trend_down", 0):
                block = cfg.get("long", {})
            else:
                block = cfg.get("short", {})
        else:
            block = cfg

        prefer = block.get("prefer", [])
        for candidate in prefer:
            if not available or candidate in available:
                LOG.info(f"[RegimeRules] regime={r} → {candidate}")
                return candidate

        return None

    def _normalize_regime_probs(self, regime_probs: dict):
        if not isinstance(regime_probs, dict):
            return {}

        out = {}
        for key, value in regime_probs.items():
            if not isinstance(value, (int, float)):
                continue
            name = str(key)
            if name.startswith("p_regime_"):
                name = name.replace("p_regime_", "", 1)
            elif name.startswith("regime_"):
                name = name.replace("regime_", "", 1)
            out[name] = float(value)
        return out
