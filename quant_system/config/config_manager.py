"""
Central configuration manager providing global, cached, structured access
to all YAML configs loaded by ConfigLoader.
"""

import threading
from typing import Any, Dict

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import log


class ConfigManager:
    """
    Singleton-style configuration manager.
    Loads all configs once and exposes structured access.
    """

    _instance_lock = threading.Lock()
    _instance = None

    def __new__(cls, conf_dir: str, env: str = None):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, conf_dir: str, env: str = None):
        if hasattr(self, "_initialized") and self._initialized:
            return  # initialized already

        self.conf_dir = conf_dir
        self.env = env
        loader = ConfigLoader(conf_dir, env)
        self._cfg = loader.load()

        self._initialized = True
        log("ConfigManager initialized with unified configuration.")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Return a top-level config section (e.g. 'models', 'features', 'execution').
        """
        return self._cfg.get(key, default)

    def get_model_cfg(self, model_name: str) -> Dict[str, Any]:
        """
        Return model config for specialist, meta, confluence, or hazard models.
        """
        models = self._cfg.get("models", {})
        if model_name not in models:
            raise KeyError(f"Model config not found for: {model_name}")
        return models[model_name]

    def get_feature_cfg(self, feature_group: str) -> Dict[str, Any]:
        """
        Return feature group config (e.g. 'smc', 'ema', 'regime', 'volatility').
        """
        features = self._cfg.get("features", {})
        if feature_group not in features:
            raise KeyError(f"Feature config not found for: {feature_group}")
        return features[feature_group]

    def get_label_cfg(self, label_name: str) -> Dict[str, Any]:
        """
        Return label generation config (e.g. 'liq_flow', 'bos_cont', 'hazard').
        """
        labels = self._cfg.get("labels", {})
        if label_name not in labels:
            raise KeyError(f"Label config not found for: {label_name}")
        return labels[label_name]

    def execution(self) -> Dict[str, Any]:
        """
        Return execution/trade-evaluator/tier rules configuration.
        """
        return self._cfg.get("execution", {})

    def reload(self):
        """
        Optional: explicit reload. Not used at runtime normally.
        """
        loader = ConfigLoader(self.conf_dir, self.env)
        self._cfg = loader.load()
        log("Configuration reloaded.")

    @property
    def full(self) -> Dict[str, Any]:
        """
        Return full resolved configuration dictionary.
        """
        return self._cfg
