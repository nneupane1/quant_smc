"""
Registry for saving/loading all ML models, hazard models, configs, and metrics.
"""

import os
import json
from pathlib import Path
import time
import joblib
from typing import Any, Dict, List, Optional, Tuple

from quant_system.utils.logger import log


class ModelRegistry:
    """
    Handles persistence of specialist models, calibrators, meta-model,
    confluence model, and hazard survival models.
    """

    def __init__(self, base_dir: str):
        self.base_dir = str(Path(base_dir))
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_manifest = os.path.join(self.base_dir, "active_models.json")
        log(f"ModelRegistry at {self.base_dir} initialized.")

    def _model_dir(self, model_name: str, version: str) -> str:
        d = os.path.join(self.base_dir, model_name, version)
        os.makedirs(d, exist_ok=True)
        return d

    def _version_dir(self, model_name: str, version: str) -> str:
        d = os.path.join(self.base_dir, model_name, version)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"No model version found: {model_name}/{version}")
        return d

    @staticmethod
    def _safe_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _version_dirs(self, model_name: str) -> List[str]:
        model_root = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_root):
            return []
        return sorted(
            [
                name
                for name in os.listdir(model_root)
                if os.path.isdir(os.path.join(model_root, name))
            ]
        )

    def _latest_model_dir(self, model_name: str) -> str:
        model_root = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_root):
            raise FileNotFoundError(f"No model found: {model_name}")
        versions = self._version_dirs(model_name)
        if not versions:
            raise FileNotFoundError(f"No versions for model: {model_name}")
        return os.path.join(model_root, versions[-1])

    @staticmethod
    def _score_payload(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[int, float, float]:
        higher_is_better = ("cv_score", "cv_ap", "pr_auc", "cv_auc", "auc")
        lower_is_better = ("brier", "logloss")
        for rank, key in enumerate(higher_is_better):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return (100 - rank, float(value), 0.0)
        for rank, key in enumerate(lower_is_better):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return (10 - rank, -float(value), 0.0)
        # Fall back to config hints if metrics are sparse.
        for rank, key in enumerate(("cv_score", "decision_threshold")):
            value = cfg.get(key)
            if isinstance(value, (int, float)):
                return (1 - rank, float(value), 0.0)
        return (0, float("-inf"), 0.0)

    def best_version(self, model_name: str) -> Tuple[str, Dict[str, Any]]:
        versions = self._version_dirs(model_name)
        if not versions:
            raise FileNotFoundError(f"No versions for model: {model_name}")

        ranked: List[Tuple[Tuple[int, float, float, str], str, Dict[str, Any]]] = []
        for version in versions:
            d = os.path.join(self.base_dir, model_name, version)
            metrics = self._safe_json(os.path.join(d, "metrics.json"))
            cfg = self._safe_json(os.path.join(d, "config.json"))
            priority, score, _aux = self._score_payload(metrics, cfg)
            ranked.append(
                (
                    (priority, score, os.path.getmtime(d), version),
                    version,
                    {"metrics": metrics, "config": cfg, "score_priority": priority, "score_value": score},
                )
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        _sort_key, version, meta = ranked[0]
        return version, meta

    def load_bundle(self, model_name: str, version: str) -> Tuple[Any, Any, Dict[str, Any]]:
        d = self._version_dir(model_name, version)
        clf = joblib.load(os.path.join(d, "clf.joblib"))
        cal_path = os.path.join(d, "cal.joblib")
        cal = joblib.load(cal_path) if os.path.exists(cal_path) else None
        cfg = self._safe_json(os.path.join(d, "config.json"))
        log(f"Loaded bundle {model_name} version={version}.")
        return clf, cal, cfg

    def load_best_bundle(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
        version, meta = self.best_version(model_name)
        clf, cal, cfg = self.load_bundle(model_name, version)
        return clf, cal, cfg, {
            "model_id": model_name,
            "version": version,
            "selection_source": "best",
            "selected_score": meta.get("score_value"),
            "selected_priority": meta.get("score_priority"),
        }

    def load_preferred_bundle(
        self,
        model_name: str,
        *,
        requested_model: Optional[str] = None,
        slot: str = "production",
    ) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
        route = self.get_active_route(requested_model or model_name, slot=slot)
        if isinstance(route, dict):
            active_model = str(route.get("model_id") or model_name)
            active_version = str(route.get("version") or "").strip()
            if active_model:
                try:
                    if not active_version:
                        active_version, _meta = self.best_version(active_model)
                    clf, cal, cfg = self.load_bundle(active_model, active_version)
                    meta = dict(route)
                    meta.setdefault("model_id", active_model)
                    meta.setdefault("version", active_version)
                    meta["selection_source"] = "active_slot"
                    return clf, cal, cfg, meta
                except Exception:
                    pass
        return self.load_best_bundle(model_name)

    def save_model(
        self,
        model_name: str,
        version: str,
        clf: Any,
        cal: Any,
        config: Dict[str, Any]
    ):
        d = self._model_dir(model_name, version)

        joblib.dump(clf, os.path.join(d, "clf.joblib"))
        if cal is not None:
            joblib.dump(cal, os.path.join(d, "cal.joblib"))

        payload = {"model_name": model_name, "version": version, **(config or {})}
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(payload, f, indent=2)

        log(f"Saved model {model_name} version={version}.")

    def save_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float]
    ):
        d = self._model_dir(model_name, version)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"Saved metrics for {model_name} version={version}.")

    def load_latest(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load latest specialist/meta/confluence model + calibrator.
        """
        d = self._latest_model_dir(model_name)

        clf = joblib.load(os.path.join(d, "clf.joblib"))

        cal_path = os.path.join(d, "cal.joblib")
        cal = joblib.load(cal_path) if os.path.exists(cal_path) else None

        log(f"Loaded latest {model_name} version={os.path.basename(d)}.")
        return clf, cal

    def load_latest_bundle(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any]]:
        d = self._latest_model_dir(model_name)
        clf = joblib.load(os.path.join(d, "clf.joblib"))
        cal_path = os.path.join(d, "cal.joblib")
        cal = joblib.load(cal_path) if os.path.exists(cal_path) else None
        cfg_path = os.path.join(d, "config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        log(f"Loaded latest bundle {model_name} version={os.path.basename(d)}.")
        return clf, cal, cfg

    def save_hazard_model(
        self,
        model_name: str,
        version: str,
        models: Dict[int, Any],
        config: Dict[str, Any]
    ):
        d = self._model_dir(model_name, version)

        # Save each bin model
        haz_dir = os.path.join(d, "hazard_bins")
        os.makedirs(haz_dir, exist_ok=True)

        for b, clf in models.items():
            joblib.dump(clf, os.path.join(haz_dir, f"bin_{b}.joblib"))

        payload = {"model_name": model_name, "version": version, **(config or {})}
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(payload, f, indent=2)

        log(f"Hazard model saved for {model_name} version={version}.")

    def load_hazard_model(
        self,
        model_name: str
    ) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """
        Load latest hazard survival model (per-bin logistic).
        """
        model_root = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_root):
            raise FileNotFoundError(f"No hazard model found: {model_name}")

        versions = sorted(os.listdir(model_root))
        latest = versions[-1]
        d = os.path.join(model_root, latest)

        with open(os.path.join(d, "config.json"), "r") as f:
            cfg = json.load(f)

        H = cfg.get("horizon_bars", 48)
        haz_dir = os.path.join(d, "hazard_bins")

        models = {}
        for b in range(1, H + 1):
            path = os.path.join(haz_dir, f"bin_{b}.joblib")
            if os.path.exists(path):
                models[b] = joblib.load(path)

        log(f"Loaded hazard model {model_name} version={latest}.")
        return models, cfg

    def set_active_model(self, model_id: str, slot: str = "production") -> Dict[str, Any]:
        manifest = self.get_active_models()
        slot_payload = manifest.get(slot)
        if not isinstance(slot_payload, dict):
            slot_payload = {}
        slot_payload["model_id"] = model_id
        slot_payload["updated_at"] = time.time()
        manifest[slot] = slot_payload
        with open(self.active_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        return slot_payload

    def set_active_route(
        self,
        requested_model: str,
        model_id: str,
        *,
        version: Optional[str] = None,
        slot: str = "production",
        selected_score: Optional[float] = None,
        selected_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        if version is None:
            version, meta = self.best_version(model_id)
            if selected_score is None:
                selected_score = meta.get("score_value")
            if selected_metric is None:
                metrics = meta.get("metrics", {})
                for key in ("cv_score", "cv_ap", "pr_auc", "cv_auc", "auc", "brier", "logloss"):
                    if isinstance(metrics.get(key), (int, float)):
                        selected_metric = key
                        break

        manifest = self.get_active_models()
        slot_payload = manifest.get(slot)
        if not isinstance(slot_payload, dict):
            slot_payload = {}
        routes = slot_payload.get("routes")
        if not isinstance(routes, dict):
            routes = {}
        entry = {
            "requested_model": str(requested_model),
            "model_id": str(model_id),
            "version": str(version),
            "updated_at": time.time(),
        }
        if selected_score is not None:
            entry["selected_score"] = float(selected_score)
        if selected_metric:
            entry["selected_metric"] = str(selected_metric)
        routes[str(requested_model)] = entry
        slot_payload["routes"] = routes
        slot_payload["updated_at"] = time.time()
        manifest[slot] = slot_payload
        with open(self.active_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        return entry

    def promote_best_version(
        self,
        requested_model: str,
        *,
        model_id: Optional[str] = None,
        slot: str = "production",
    ) -> Dict[str, Any]:
        target_model = str(model_id or requested_model)
        version, meta = self.best_version(target_model)
        metrics = meta.get("metrics", {})
        selected_metric = None
        for key in ("cv_score", "cv_ap", "pr_auc", "cv_auc", "auc", "brier", "logloss"):
            if isinstance(metrics.get(key), (int, float)):
                selected_metric = key
                break
        return self.set_active_route(
            requested_model,
            target_model,
            version=version,
            slot=slot,
            selected_score=meta.get("score_value"),
            selected_metric=selected_metric,
        )

    def get_active_models(self) -> Dict[str, Any]:
        if not os.path.exists(self.active_manifest):
            return {}
        try:
            with open(self.active_manifest, "r") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def get_active_model(self, slot: str = "production") -> Any:
        return self.get_active_models().get(slot)

    def get_active_route(self, requested_model: str, slot: str = "production") -> Optional[Dict[str, Any]]:
        slot_payload = self.get_active_model(slot)
        if not isinstance(slot_payload, dict):
            return None
        routes = slot_payload.get("routes")
        if not isinstance(routes, dict):
            return None
        entry = routes.get(str(requested_model))
        return entry if isinstance(entry, dict) else None
