"""
weight_scheduler.py
Computes ensemble weights based on regime, session, volatility or custom rules.
Supports:
 • static weights
 • regime-conditioned weights
 • smoothing
"""

import numpy as np


class WeightScheduler:

    def __init__(self, config: dict):
        self.config = config or {}
        self.static = (
            self.config.get("static_weights")
            or self._from_simple_weights(self.config.get("weights", {}), key="default")
            or {}
        )
        self.regime_map = (
            self.config.get("regime_weights")
            or self._from_simple_weights(self.config.get("weights", {}))
            or {}
        )
        self.smooth = self.config.get("smoothing", 0.1)
        self._last_weights = {}

    # --------------------------------------------------------------
    def compute_weights(self, regime_probs: dict, model_names=None):
        """
        Blend:
         • static weights
         • regime-conditioned weights
         • normalization + smoothing
        """

        model_names = list(model_names or [])
        w = dict(self.static)
        if not w and model_names:
            w = {name: 1.0 / len(model_names) for name in model_names}
        for name in model_names:
            w.setdefault(name, 0.0)

        # Add regime-conditioned adjustments
        # Example in config:
        # regime_weights:
        #    trend:
        #        model_A: +0.2
        #        model_B: -0.1
        #
        for regime, p in regime_probs.items():
            if regime in self.regime_map:
                for m, delta in self.regime_map[regime].items():
                    w[m] = w.get(m, 0) + p * delta

        # Rectify negatives
        for k in w:
            w[k] = max(w[k], 0)

        # Normalize
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        elif model_names:
            w = {name: 1.0 / len(model_names) for name in model_names}

        # Smooth weights (EMA smoothing)
        smoothed = {}
        for k, v in w.items():
            prev = self._last_weights.get(k, self.static.get(k, v))
            smoothed[k] = prev * self.smooth + v * (1 - self.smooth)
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        self._last_weights = dict(smoothed)

        return smoothed

    # --------------------------------------------------------------
    def _from_simple_weights(self, weights_cfg: dict, key: str = None):
        """
        Helper to support simplified weights section:
          weights:
            default:
              model_a: 0.6
              model_b: 0.4
            trend:
              model_a: 0.7
              model_b: 0.3
        """
        if not weights_cfg:
            return {}
        if key:
            return weights_cfg.get(key, {})
        # build regime_map deltas (as absolute weights)
        regime_map = {}
        for regime, mp in weights_cfg.items():
            if regime == "default":
                continue
            regime_map[regime] = {m: float(w) for m, w in mp.items()}
        return regime_map
