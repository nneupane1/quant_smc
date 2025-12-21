"""
Predictor for specialist models, meta-model, confluence, and hazard.
Uses ModelRegistry artifacts (clf.joblib + optional cal.joblib).
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from quant_system.utils.logger import log
from quant_system.ml.registry.model_registry import ModelRegistry


class ModelPredictor:
    """
    Loads trained models from registry and computes:
        - Specialist probabilities
        - Meta-model probability
        - Confluence probability
        - Hazard curve (per-bin survival)
        - Hazard score
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        log("ModelPredictor initialized.")

    def _predict_specialist(self, model_name: str, x: np.ndarray) -> float:
        clf, cal = self.registry.load_latest(model_name)
        if isinstance(clf, dict) and "model" in clf:
            base = clf["model"]
            p = base.predict_proba(x.reshape(1, -1))[0][1]
        else:
            p = clf.predict_proba(x.reshape(1, -1))[0][1]
        if cal:
            if hasattr(cal, "predict_proba"):
                p = cal.predict_proba([[p]])[0][1]
            elif hasattr(cal, "predict"):
                p = cal.predict([p])[0]
        return float(p)

    def _predict_hazard_curve(self, X: np.ndarray, model_name: str) -> Dict[int, float]:
        """
        Returns per-bin event probability.
        """
        models, cfg = self.registry.load_hazard_model(model_name)
        H = cfg.get("horizon_bars", 48)
        event_probs = {}

        for b in range(1, H + 1):
            if b in models:
                clf = models[b]
                p = clf.predict_proba(X.reshape(1, -1))[0][1]
                event_probs[b] = float(p)
            else:
                event_probs[b] = 0.0

        return event_probs

    def _hazard_score(self, curve: Dict[int, float]) -> float:
        """
        Converts hazard curve into a single scalar score.
        """
        score = 0.0
        denom = 0.0
        for b, p in curve.items():
            w = 1.0 / b
            score += w * p
            denom += w
        if denom == 0:
            return 0.0
        return score / denom

    def predict_single(self, x_row: List[float], specialist_list: List[str]) -> Dict[str, Any]:
        """
        Compute full prediction set for a single feature row.
        """
        X = np.array(x_row, dtype=float)
        predictions = {}

        # Specialist models
        specialist_probs = []
        for model_name in specialist_list:
            p = self._predict_specialist(model_name, X)
            predictions[f"prob_{model_name}"] = p
            specialist_probs.append(p)

        # Meta-model
        try:
            clf_meta, _ = self.registry.load_latest("meta_model")
            specialist_vec = np.array(specialist_probs, dtype=float).reshape(1, -1)
            p_meta = clf_meta.predict_proba(specialist_vec)[0][1]
            predictions["prob_meta"] = float(p_meta)
        except Exception as e:
            predictions["prob_meta"] = None

        # Confluence model
        try:
            clf_conf, _ = self.registry.load_latest("confluence_model")
            specialist_vec = np.array(specialist_probs, dtype=float).reshape(1, -1)
            p_conf = clf_conf.predict_proba(specialist_vec)[0][1]
            predictions["prob_confluence"] = float(p_conf)
        except Exception:
            predictions["prob_confluence"] = None

        # Hazard
        try:
            hazard_curve = self._predict_hazard_curve(X, "hazard")
            predictions["hazard_curve"] = hazard_curve
            predictions["hazard_score"] = self._hazard_score(hazard_curve)
        except Exception:
            predictions["hazard_curve"] = {}
            predictions["hazard_score"] = None

        return predictions

    def predict_batch(self, X: List[List[float]], specialist_list: List[str]) -> List[Dict[str, Any]]:
        """
        Predict batch.
        """
        out = []
        for i, row in enumerate(X):
            if i % 5000 == 0:
                log(f"Predicting row {i}...")
            out.append(self.predict_single(row, specialist_list))
        log("Batch prediction complete.")
        return out
