"""
consensus_engine.py
Multi-Model Consensus Engine
Combines multiple specialist model versions into a unified probability vector.

Supports:
 • Weighted voting
 • Softmax averaging
 • Regime-conditioned weighting
 • Confidence filtering
 • Disagreement flags (for hazard and risk tightening)
"""

import numpy as np
from typing import Any, Dict
from quant_system.utils.logger import get_logger
from quant_system.model_ensemble.weight_scheduler import WeightScheduler
from quant_system.model_ensemble.disagreement_detector import DisagreementDetector

LOG = get_logger("consensus_engine")


class ConsensusEngine:
    """
    Collects predictions from N model versions and outputs:
      • fused probabilities
      • disagreement metrics
      • confidence score
      • which model dominated (if any)
    """

    def __init__(self, model_versions: dict, config: dict, regime_provider):
        """
        model_versions : { name: predictor_instance }
        config : from models.yaml
        regime_provider : class with method get_regime_state()
        """
        self.models = model_versions
        self.config = config or {}
        ensemble_cfg = self.config.get("ensemble", self.config)
        self.regime_provider = regime_provider

        self.scheduler = WeightScheduler(ensemble_cfg)
        self.disagreement = DisagreementDetector()
        self.output_keys = list(ensemble_cfg.get("output_keys", []))

        LOG.info(f"[Consensus] Loaded ensemble models: {list(self.models.keys())}")

    # ------------------------------------------------------------------
    def predict(self, features: dict) -> dict:
        """
        Returns:
        {
          "probs": { specialist: fused_prob },
          "model_contrib": { name: weight },
          "disagreement": {...},
          "confidence": float
        }
        """

        regime = self._get_regime_state()
        weights = self.scheduler.compute_weights(regime, model_names=list(self.models.keys()))

        raw_outputs = {}
        for name, model in self.models.items():
            try:
                raw_outputs[name] = self._predict_one(model, features)
            except Exception as e:
                LOG.error(f"[Consensus] Model {name} failed: {e}")
                raw_outputs[name] = None

        fused = self._fuse(raw_outputs, weights)

        disagree_stats = self.disagreement.evaluate(raw_outputs)

        return {
            "probs": fused,
            "model_contrib": weights,
            "disagreement": disagree_stats,
            "confidence": max(0.0, 1.0 - disagree_stats.get("spread", 0))
        }

    # ------------------------------------------------------------------
    def _get_regime_state(self) -> Dict[str, float]:
        provider = self.regime_provider
        if provider is None:
            return {}
        if isinstance(provider, dict):
            return {
                str(k): float(v)
                for k, v in provider.items()
                if isinstance(v, (int, float, np.number))
            }
        if hasattr(provider, "get_regime_state"):
            regime = provider.get_regime_state()
            if isinstance(regime, dict):
                return {
                    str(k): float(v)
                    for k, v in regime.items()
                    if isinstance(v, (int, float, np.number))
                }
        return {}

    def _predict_one(self, model: Any, features: dict) -> Dict[str, float]:
        if hasattr(model, "predict_single"):
            outputs = model.predict_single(features, [])
        elif hasattr(model, "predict"):
            outputs = model.predict(features)
        elif callable(model):
            outputs = model(features)
        else:
            raise TypeError(f"Unsupported ensemble model type: {type(model)!r}")

        if not isinstance(outputs, dict):
            raise TypeError("Ensemble model outputs must be dict-like.")

        selected = self._select_outputs(outputs)
        return selected

    def _select_outputs(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        if self.output_keys:
            return {
                key: float(outputs[key])
                for key in self.output_keys
                if key in outputs and isinstance(outputs[key], (int, float, np.number))
            }

        selected = {}
        for key, value in outputs.items():
            if not isinstance(value, (int, float, np.number)):
                continue
            if key.startswith("prob_") or key in {"hazard_score", "cvar"}:
                selected[key] = float(value)
        return selected

    # ------------------------------------------------------------------
    def _fuse(self, raw_outputs, weights):
        """
        Weighted averaging with softmax normalization.
        raw_outputs format:
           { model_name : {specialist: prob, specialist2: prob2, ... } }
        """
        specialists = set()
        for o in raw_outputs.values():
            if o:
                specialists.update(o.keys())
        specialists = sorted(list(specialists))

        fused = {k: 0.0 for k in specialists}
        total_weight = 0.0

        # Weighted sum
        for model_name, output in raw_outputs.items():
            if not output:
                continue
            w = float(weights.get(model_name, 0.0))
            if w <= 0:
                continue
            total_weight += w
            for sp in specialists:
                fused[sp] += w * float(output.get(sp, 0.0))

        if total_weight > 0:
            fused = {k: v / total_weight for k, v in fused.items()}
        return fused
