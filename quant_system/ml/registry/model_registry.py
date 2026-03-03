"""
Registry for saving/loading all ML models, hazard models, configs, and metrics.
"""

import os
import json
from pathlib import Path
import time
import joblib
from typing import Any, Dict, Tuple

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

    def _latest_model_dir(self, model_name: str) -> str:
        model_root = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_root):
            raise FileNotFoundError(f"No model found: {model_name}")
        versions = sorted(
            [name for name in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, name))]
        )
        if not versions:
            raise FileNotFoundError(f"No versions for model: {model_name}")
        return os.path.join(model_root, versions[-1])

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
        manifest[slot] = {
            "model_id": model_id,
            "updated_at": time.time(),
        }
        with open(self.active_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest[slot]

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
