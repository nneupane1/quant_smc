"""
Predictor for specialist models, meta-model, confluence, and hazard.
Uses ModelRegistry artifacts (clf.joblib + optional cal.joblib).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

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

    SPECIALIST_MODELS = {"liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"}

    def __init__(self, registry: ModelRegistry, *, prefer_tcn_specialists: bool = True):
        self.registry = registry
        self.prefer_tcn_specialists = bool(prefer_tcn_specialists)
        self._resolved_specialist_names: Dict[str, str] = {}
        log(f"ModelPredictor initialized. prefer_tcn_specialists={self.prefer_tcn_specialists}")

    def source_mode(self) -> str:
        return "tcn_first" if self.prefer_tcn_specialists else "tree_first"

    def specialist_source_map(self) -> Dict[str, str]:
        return dict(self._resolved_specialist_names)

    @staticmethod
    def _is_asset_specific(name: str) -> bool:
        return "_" in name and name.split("_", 1)[0].isupper()

    def _candidate_specialist_names(self, model_name: str) -> List[str]:
        name = str(model_name)
        if name.endswith("_tcn"):
            return [name]
        if self._is_asset_specific(name):
            return [name]
        if name not in self.SPECIALIST_MODELS:
            return [name]

        # Keep canonical downstream naming (prob_liq_flow etc.) while allowing
        # model source routing to prefer TCN artifacts by default.
        if self.prefer_tcn_specialists:
            return [f"{name}_tcn", name, f"BTCUSD_{name}_tcn", f"BTCUSD_{name}"]
        return [name, f"{name}_tcn", f"BTCUSD_{name}", f"BTCUSD_{name}_tcn"]

    def _load_specialist_bundle(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any], str]:
        requested = str(model_name)
        cached = self._resolved_specialist_names.get(requested)
        if cached is not None:
            try:
                clf, cal, cfg = self.registry.load_latest_bundle(cached)
                return clf, cal, cfg, cached
            except Exception:
                # stale cache entry (artifact rotation/deletion) -> re-resolve
                self._resolved_specialist_names.pop(requested, None)

        last_exc: Optional[Exception] = None
        for candidate in self._candidate_specialist_names(requested):
            try:
                clf, cal, cfg = self.registry.load_latest_bundle(candidate)
                self._resolved_specialist_names[requested] = candidate
                if candidate != requested:
                    log(f"ModelPredictor specialist route: {requested} -> {candidate}")
                return clf, cal, cfg, candidate
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"No model candidate resolved for specialist '{requested}'.")

    def warmup_specialists(self, specialist_list: Optional[List[str]] = None) -> Dict[str, str]:
        names = specialist_list or sorted(self.SPECIALIST_MODELS)
        for model_name in names:
            try:
                _clf, _cal, _cfg, _resolved = self._load_specialist_bundle(model_name)
            except Exception:
                continue
        return self.specialist_source_map()

    @staticmethod
    def _row_frame(row_like, feature_cols: List[str]):
        if isinstance(row_like, (list, tuple, np.ndarray)):
            return np.asarray(row_like, dtype=float).reshape(1, -1)
        if not feature_cols:
            numeric_vals = [v for _, v in dict(row_like).items() if isinstance(v, (int, float, np.number))]
            return np.asarray(numeric_vals, dtype=float).reshape(1, -1)
        row = row_like if isinstance(row_like, dict) else dict(row_like)
        data = {col: [row.get(col, np.nan)] for col in feature_cols}
        return pd.DataFrame(data)

    @staticmethod
    def _positive_class_proba(model: Any, X_in) -> float:
        proba = model.predict_proba(X_in)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            return float(arr[0])
        if arr.shape[1] == 1:
            classes_attr = getattr(model, "classes_", None)
            classes = list(classes_attr) if classes_attr is not None else []
            if classes == [1]:
                return 1.0
            return 0.0
        classes_attr = getattr(model, "classes_", None)
        classes = list(classes_attr) if classes_attr is not None else []
        if 1 in classes:
            return float(arr[0, classes.index(1)])
        return float(arr[0, min(1, arr.shape[1] - 1)])

    def _predict_specialist(self, model_name: str, row_like) -> float:
        clf, cal, cfg, _resolved_name = self._load_specialist_bundle(model_name)
        feature_cols = cfg.get("features", [])
        X_in = self._row_frame(row_like, feature_cols) if feature_cols else self._row_frame(row_like, [])
        if isinstance(clf, dict) and "model" in clf:
            base = clf["model"]
            p = self._positive_class_proba(base, X_in)
        else:
            p = self._positive_class_proba(clf, X_in)
        if cal:
            if hasattr(cal, "predict_proba"):
                p = cal.predict_proba([[p]])[0][1]
            elif hasattr(cal, "predict"):
                p = cal.predict([p])[0]
        return float(p)

    def _predict_stack(self, model_name: str, specialist_probs: Dict[str, float]) -> Optional[float]:
        clf, _cal, cfg = self.registry.load_latest_bundle(model_name)
        stack_inputs = cfg.get("stack_inputs", list(specialist_probs.keys()))
        if not stack_inputs:
            return None
        vec = np.array([float(specialist_probs.get(name, 0.0)) for name in stack_inputs], dtype=float).reshape(1, -1)
        return self._positive_class_proba(clf, vec)

    def _predict_hazard_curve(self, row_like, model_name: str) -> Dict[int, float]:
        """
        Returns per-bin event probability.
        """
        models, cfg = self.registry.load_hazard_model(model_name)
        H = cfg.get("horizon_bars", 48)
        event_probs = {}
        feature_cols = cfg.get("feature_cols", [])
        X_in = self._row_frame(row_like, feature_cols) if feature_cols else self._row_frame(row_like, [])

        for b in range(1, H + 1):
            if b in models:
                clf = models[b]
                p = self._positive_class_proba(clf, X_in)
                event_probs[b] = float(p)
            else:
                event_probs[b] = 0.0

        return event_probs

    def _predict_quantiles(self, row_like, model_name: str = "quantile") -> Dict[str, float]:
        models, _cal, cfg = self.registry.load_latest_bundle(model_name)
        feature_cols = cfg.get("features", [])
        X_in = self._row_frame(row_like, feature_cols) if feature_cols else self._row_frame(row_like, [])
        quantiles = {}
        if isinstance(models, dict):
            for key, model in models.items():
                try:
                    raw = str(key).lower().replace("q_", "").replace("q", "")
                    q_float = float(raw)
                    q = f"q{int(round(q_float * 100)):02d}"
                    quantiles[q] = float(model.predict(X_in)[0])
                except Exception:
                    continue
        return quantiles

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

    def predict_single(self, x_row: Any, specialist_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute full prediction set for a single feature row.
        """
        predictions = {}
        specialist_list = specialist_list or []

        # Specialist models
        specialist_probs: Dict[str, float] = {}
        for model_name in specialist_list:
            p = self._predict_specialist(model_name, x_row)
            predictions[f"prob_{model_name}"] = p
            specialist_probs[model_name] = p

        # Meta-model
        try:
            if specialist_probs:
                predictions["prob_meta"] = self._predict_stack("meta_model", specialist_probs)
            else:
                predictions["prob_meta"] = None
        except Exception:
            predictions["prob_meta"] = None

        # Confluence model
        try:
            if specialist_probs:
                predictions["prob_confluence"] = self._predict_stack("confluence_model", specialist_probs)
            else:
                predictions["prob_confluence"] = None
        except Exception:
            predictions["prob_confluence"] = None

        # Hazard
        try:
            hazard_curve = self._predict_hazard_curve(x_row, "hazard")
            predictions["hazard_curve"] = hazard_curve
            predictions["hazard_score"] = self._hazard_score(hazard_curve)
        except Exception:
            predictions["hazard_curve"] = {}
            predictions["hazard_score"] = None

        # Quantiles
        try:
            quantiles = self._predict_quantiles(x_row, "quantile")
            predictions["quantiles"] = quantiles
            for key, value in quantiles.items():
                predictions[key] = value
            if quantiles:
                q05 = float(quantiles.get("q05", quantiles.get("q0.05", 0.0)) or 0.0)
                q10 = float(quantiles.get("q10", quantiles.get("q0.10", 0.0)) or 0.0)
                predictions["cvar"] = abs(min((q05 + q10) / 2.0, 0.0))
        except Exception:
            predictions["quantiles"] = {}

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
