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
        self.regime_provider = regime_provider

        self.scheduler = WeightScheduler(self.config.get("ensemble", {}))
        self.disagreement = DisagreementDetector()

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

        regime = self.regime_provider.get_regime_state() if self.regime_provider else {}
        weights = self.scheduler.compute_weights(regime)

        raw_outputs = {}
        for name, model in self.models.items():
            try:
                raw_outputs[name] = model.predict(features)
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

        # Weighted sum
        for model_name, output in raw_outputs.items():
            if not output:
                continue
            w = weights.get(model_name, 0)
            for sp in specialists:
                fused[sp] += w * output.get(sp, 0)

        # Normalize to 0…1 softmax-like vector
        values = np.array(list(fused.values()))
        if values.sum() > 0:
            values = values / values.sum()

        return dict(zip(specialists, values))
