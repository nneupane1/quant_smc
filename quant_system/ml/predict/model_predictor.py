"""
Predictor for specialist models, meta-model, confluence, and hazard.
Uses ModelRegistry artifacts (clf.joblib + optional cal.joblib).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from quant_system.utils.logger import log
from quant_system.ml.registry.model_registry import ModelRegistry


INFERENCE_ROUTING_MODES = {"tree", "tcn", "hybrid_explicit"}


def _normalize_routing_mode(value: Any, *, default: str) -> str:
    raw = str(value or "").strip().lower()
    return raw if raw in INFERENCE_ROUTING_MODES else default


def resolve_inference_preference(pref_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pref = pref_cfg if isinstance(pref_cfg, dict) else {}
    requested = pref.get("routing_mode")
    legacy = pref.get("prefer_tcn_specialists")
    if requested is None:
        if isinstance(legacy, bool):
            requested = "tcn" if legacy else "tree"
        else:
            requested = "tree"

    routing_mode = _normalize_routing_mode(requested, default="tree")
    challenger_mode = _normalize_routing_mode(pref.get("challenger_mode", "tcn"), default="tcn")
    allow_hybrid_explicit = bool(pref.get("allow_hybrid_explicit", False))
    active_slot = str(pref.get("active_slot", "production") or "production").strip() or "production"

    return {
        "routing_mode": routing_mode,
        "challenger_mode": challenger_mode,
        "allow_hybrid_explicit": allow_hybrid_explicit,
        "active_slot": active_slot,
    }


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
    STACK_MODELS = {"meta_model", "confluence_model"}

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        routing_mode: str = "tree",
        challenger_mode: str = "tcn",
        allow_hybrid_explicit: bool = False,
        active_slot: str = "production",
        prefer_tcn_specialists: Optional[bool] = None,
    ):
        self.registry = registry
        if prefer_tcn_specialists is not None:
            routing_mode = "tcn" if bool(prefer_tcn_specialists) else "tree"
        self.requested_routing_mode = _normalize_routing_mode(routing_mode, default="tree")
        self.challenger_routing_mode = _normalize_routing_mode(challenger_mode, default="tcn")
        self.allow_hybrid_explicit = bool(allow_hybrid_explicit)
        self.active_slot = str(active_slot or "production")
        self._resolved_specialist_names: Dict[str, str] = {}
        self._resolved_stack_names: Dict[str, str] = {}
        self._resolved_specialist_meta: Dict[str, Dict[str, Any]] = {}
        self._resolved_stack_meta: Dict[str, Dict[str, Any]] = {}
        self._specialist_bundle_cache: Dict[str, Tuple[Any, Any, Dict[str, Any], str]] = {}
        self._stack_bundle_cache: Dict[str, Tuple[Any, Any, Dict[str, Any], str]] = {}
        self._hazard_bundle_cache: Dict[str, Tuple[Dict[int, Any], Dict[str, Any]]] = {}
        self._quantile_bundle_cache: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {}
        self._routing_note = ""
        self._effective_routing_mode = self._resolve_effective_routing_mode()
        log(
            "ModelPredictor initialized. "
            f"requested_route={self.requested_routing_mode} "
            f"effective_route={self._effective_routing_mode} "
            f"challenger={self.challenger_routing_mode} "
            f"active_slot={self.active_slot} "
            f"allow_hybrid_explicit={self.allow_hybrid_explicit} "
            f"note={self._routing_note or 'ready'}"
        )

    def source_mode(self) -> str:
        return self._effective_routing_mode

    def routing_status(self) -> Dict[str, Any]:
        return {
            "requested_mode": self.requested_routing_mode,
            "effective_mode": self._effective_routing_mode,
            "challenger_mode": self.challenger_routing_mode,
            "active_slot": self.active_slot,
            "allow_hybrid_explicit": self.allow_hybrid_explicit,
            "note": self._routing_note,
        }

    def specialist_source_map(self) -> Dict[str, str]:
        return dict(self._resolved_specialist_names)

    def stack_source_map(self) -> Dict[str, str]:
        return dict(self._resolved_stack_names)

    def specialist_bundle_map(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._resolved_specialist_meta.items()}

    def stack_bundle_map(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._resolved_stack_meta.items()}

    @staticmethod
    def _is_asset_specific(name: str) -> bool:
        return "_" in name and name.split("_", 1)[0].isupper()

    def _model_exists(self, model_name: str) -> bool:
        model_root = Path(self.registry.base_dir) / str(model_name)
        if not model_root.exists() or not model_root.is_dir():
            return False
        return any(child.is_dir() for child in model_root.iterdir())

    def _stack_family_ready(self, suffix: str) -> bool:
        suffix = str(suffix).strip()
        required = [f"{name}_{suffix}" for name in sorted(self.STACK_MODELS)]
        return all(self._model_exists(name) or self._model_exists(f"BTCUSD_{name}") for name in required)

    def _resolve_effective_routing_mode(self) -> str:
        requested = self.requested_routing_mode
        if requested == "tree":
            return "tree"
        if requested == "hybrid_explicit":
            if not self.allow_hybrid_explicit:
                self._routing_note = "hybrid_explicit requested but disabled; falling back to tree"
                return "tree"
            if not self._stack_family_ready("hybrid"):
                self._routing_note = "hybrid stack artifacts missing; falling back to tree"
                return "tree"
            return "hybrid_explicit"
        if requested == "tcn":
            if not self._stack_family_ready("tcn"):
                self._routing_note = "tcn stack artifacts missing; falling back to tree"
                return "tree"
            return "tcn"
        self._routing_note = f"unknown routing mode '{requested}'; falling back to tree"
        return "tree"

    def _candidate_specialist_names(self, model_name: str) -> List[str]:
        name = str(model_name)
        if name.endswith("_tcn"):
            return [name]
        if self._is_asset_specific(name):
            return [name]
        if name not in self.SPECIALIST_MODELS:
            return [name]

        mode = self.source_mode()
        if mode == "tcn":
            return [f"{name}_tcn", f"BTCUSD_{name}_tcn"]
        if mode == "hybrid_explicit":
            return [f"{name}_tcn", name, f"BTCUSD_{name}_tcn", f"BTCUSD_{name}"]
        return [name, f"BTCUSD_{name}"]

    def _load_specialist_bundle(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any], str]:
        requested = str(model_name)
        cached_bundle = self._specialist_bundle_cache.get(requested)
        if cached_bundle is not None:
            return cached_bundle
        cached = self._resolved_specialist_names.get(requested)
        if cached is not None:
            try:
                cached_meta = self._resolved_specialist_meta.get(requested, {})
                version = str(cached_meta.get("version") or "").strip()
                if version:
                    clf, cal, cfg = self.registry.load_bundle(cached, version)
                else:
                    clf, cal, cfg, meta = self.registry.load_preferred_bundle(
                        cached,
                        requested_model=requested,
                        slot=self.active_slot,
                    )
                    self._resolved_specialist_meta[requested] = {
                        **meta,
                        "decision_threshold": cfg.get("decision_threshold"),
                    }
                bundle = (clf, cal, cfg, cached)
                self._specialist_bundle_cache[requested] = bundle
                return bundle
            except Exception:
                # stale cache entry (artifact rotation/deletion) -> re-resolve
                self._resolved_specialist_names.pop(requested, None)
                self._resolved_specialist_meta.pop(requested, None)
                self._specialist_bundle_cache.pop(requested, None)

        last_exc: Optional[Exception] = None
        for candidate in self._candidate_specialist_names(requested):
            try:
                clf, cal, cfg, meta = self.registry.load_preferred_bundle(
                    candidate,
                    requested_model=requested,
                    slot=self.active_slot,
                )
                resolved_name = str(meta.get("model_id") or candidate)
                self._resolved_specialist_names[requested] = resolved_name
                self._resolved_specialist_meta[requested] = {
                    **meta,
                    "decision_threshold": cfg.get("decision_threshold"),
                }
                if resolved_name != requested:
                    log(
                        "ModelPredictor specialist route: "
                        f"{requested} -> {resolved_name} "
                        f"version={meta.get('version')} "
                        f"source={meta.get('selection_source', 'best')}"
                    )
                bundle = (clf, cal, cfg, resolved_name)
                self._specialist_bundle_cache[requested] = bundle
                return bundle
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"No model candidate resolved for specialist '{requested}'.")

    def _candidate_stack_names(self, model_name: str) -> List[str]:
        name = str(model_name)
        if self._is_asset_specific(name):
            return [name]

        mode = self.source_mode()
        if mode == "tcn":
            return [f"{name}_tcn", f"BTCUSD_{name}_tcn"]
        if mode == "hybrid_explicit":
            return [f"{name}_hybrid", f"BTCUSD_{name}_hybrid"]
        return [name, f"BTCUSD_{name}"]

    def _load_stack_bundle(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any], str]:
        requested = str(model_name)
        cached_bundle = self._stack_bundle_cache.get(requested)
        if cached_bundle is not None:
            return cached_bundle
        cached = self._resolved_stack_names.get(requested)
        if cached is not None:
            try:
                cached_meta = self._resolved_stack_meta.get(requested, {})
                version = str(cached_meta.get("version") or "").strip()
                if version:
                    clf, cal, cfg = self.registry.load_bundle(cached, version)
                else:
                    clf, cal, cfg, meta = self.registry.load_preferred_bundle(
                        cached,
                        requested_model=requested,
                        slot=self.active_slot,
                    )
                    self._resolved_stack_meta[requested] = {
                        **meta,
                        "decision_threshold": cfg.get("decision_threshold"),
                    }
                bundle = (clf, cal, cfg, cached)
                self._stack_bundle_cache[requested] = bundle
                return bundle
            except Exception:
                self._resolved_stack_names.pop(requested, None)
                self._resolved_stack_meta.pop(requested, None)
                self._stack_bundle_cache.pop(requested, None)

        last_exc: Optional[Exception] = None
        for candidate in self._candidate_stack_names(requested):
            try:
                clf, cal, cfg, meta = self.registry.load_preferred_bundle(
                    candidate,
                    requested_model=requested,
                    slot=self.active_slot,
                )
                resolved_name = str(meta.get("model_id") or candidate)
                self._resolved_stack_names[requested] = resolved_name
                self._resolved_stack_meta[requested] = {
                    **meta,
                    "decision_threshold": cfg.get("decision_threshold"),
                }
                if resolved_name != requested:
                    log(
                        "ModelPredictor stack route: "
                        f"{requested} -> {resolved_name} "
                        f"version={meta.get('version')} "
                        f"source={meta.get('selection_source', 'best')}"
                    )
                bundle = (clf, cal, cfg, resolved_name)
                self._stack_bundle_cache[requested] = bundle
                return bundle
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"No model candidate resolved for stack '{requested}'.")

    def warmup_specialists(self, specialist_list: Optional[List[str]] = None) -> Dict[str, str]:
        names = specialist_list or sorted(self.SPECIALIST_MODELS)
        for model_name in names:
            try:
                _clf, _cal, _cfg, _resolved = self._load_specialist_bundle(model_name)
            except Exception:
                continue
        return self.specialist_source_map()

    def warmup_stacks(self, stack_list: Optional[List[str]] = None) -> Dict[str, str]:
        names = stack_list or sorted(self.STACK_MODELS)
        for model_name in names:
            try:
                _clf, _cal, _cfg, _resolved = self._load_stack_bundle(model_name)
            except Exception:
                continue
        return self.stack_source_map()

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
        clf, _cal, cfg, _resolved_name = self._load_stack_bundle(model_name)
        stack_inputs = cfg.get("stack_inputs", list(specialist_probs.keys()))
        if not stack_inputs:
            return None
        vec = np.array([float(specialist_probs.get(name, 0.0)) for name in stack_inputs], dtype=float).reshape(1, -1)
        return self._positive_class_proba(clf, vec)

    def _predict_hazard_curve(self, row_like, model_name: str) -> Dict[int, float]:
        """
        Returns per-bin event probability.
        """
        cached_bundle = self._hazard_bundle_cache.get(str(model_name))
        if cached_bundle is None:
            cached_bundle = self.registry.load_hazard_model(model_name)
            self._hazard_bundle_cache[str(model_name)] = cached_bundle
        models, cfg = cached_bundle
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
        cached_bundle = self._quantile_bundle_cache.get(str(model_name))
        if cached_bundle is None:
            cached_bundle = self.registry.load_latest_bundle(model_name)
            self._quantile_bundle_cache[str(model_name)] = cached_bundle
        models, _cal, cfg = cached_bundle
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
