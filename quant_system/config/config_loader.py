"""
Unified YAML configuration loader with layered merging and .env support.
Supports recursive YAML discovery under the config directory.
"""

import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv
from copy import deepcopy

from quant_system.utils.logger import log


class ConfigLoader:
    """
    Loads layered YAML configs from conf/ directory.
    Supports:
        - base.yaml
        - module-level yaml files (recursively)
        - env overrides (DEV/TEST/LIVE)
        - regime/session overrides
        - .env injection
    """

    def __init__(self, conf_dir: str, env: str = None):
        self.conf_dir = conf_dir
        self.env = env or os.getenv("QS_ENV", "DEV").upper()
        load_dotenv()
        log(f"ConfigLoader initialized. env={self.env}")

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge: override keys replace base keys recursively.
        """
        out = deepcopy(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = self._merge(out[k], v)
            else:
                out[k] = v
        return out

    def _find_file(self, filename: str) -> str:
        """
        Search recursively for a filename under conf_dir.
        """
        for root, dirs, files in os.walk(self.conf_dir):
            if filename in files:
                return os.path.join(root, filename)
        return os.path.join(self.conf_dir, filename)

    def _load_base(self) -> Dict[str, Any]:
        path = os.path.join(self.conf_dir, "base.yaml")
        cfg = self._load_yaml(path)
        log("Loaded base.yaml")
        return cfg

    # Public helper for modules that need a single YAML file
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a specific YAML file from the config directory (recursively).
        """
        path = self._find_file(filename)
        return self._load_yaml(path)

    def _load_modules(self) -> Dict[str, Any]:
        """
        Recursively load all yaml files except base and env override and overrides dirs.
        """
        merged = {}
        skip_dirs = {"regime_overrides", "session_overrides"}
        skip_files = {"base.yaml", f"{self.env}.yaml"}
        yaml_paths = []

        for root, dirs, files in os.walk(self.conf_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if fname.endswith(".yaml") and fname not in skip_files:
                    yaml_paths.append(os.path.join(root, fname))

        for path in sorted(yaml_paths):
            mod = self._load_yaml(path)
            merged = self._merge(merged, mod)
            log(f"Loaded {os.path.relpath(path, self.conf_dir)}")
        return merged

    def _load_env_override(self) -> Dict[str, Any]:
        """
        Load DEV.yaml, TEST.yaml, LIVE.yaml if present.
        """
        fname = f"{self.env}.yaml"
        path = self._find_file(fname)
        if os.path.exists(path):
            log(f"Loaded env override {fname}")
            return self._load_yaml(path)
        return {}

    def _load_regime_overrides(self) -> Dict[str, Any]:
        """
        Merge all overrides in conf/regime_overrides/
        """
        d = os.path.join(self.conf_dir, "regime_overrides")
        if not os.path.exists(d):
            return {}
        merged = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".yaml"):
                path = os.path.join(d, f)
                cfg = self._load_yaml(path)
                merged = self._merge(merged, cfg)
                log(f"Loaded regime override {f}")
        return merged

    def _load_session_overrides(self) -> Dict[str, Any]:
        """
        Merge session overrides in conf/session_overrides/
        """
        d = os.path.join(self.conf_dir, "session_overrides")
        if not os.path.exists(d):
            return {}
        merged = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".yaml"):
                path = os.path.join(d, f)
                cfg = self._load_yaml(path)
                merged = self._merge(merged, cfg)
                log(f"Loaded session override {f}")
        return merged

    def _inject_env_vars(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace any config entries that reference env vars like:
            api_key: ${KRAKEN_API_KEY}
        """
        def resolve(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                return os.getenv(key)
            return value

        def recurse(d):
            updated = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    updated[k] = recurse(v)
                else:
                    updated[k] = resolve(v)
            return updated

        cfg2 = recurse(cfg)
        log("Injected .env variables into configuration.")
        return cfg2

    def load(self) -> Dict[str, Any]:
        """
        Full resolved config.
        Order of precedence:
            base.yaml
            module yamls (recursive)
            env override
            regime overrides
            session overrides
            .env injection
        """

        cfg = self._load_base()
        cfg = self._merge(cfg, self._load_modules())
        cfg = self._merge(cfg, self._load_env_override())
        cfg = self._merge(cfg, self._load_regime_overrides())
        cfg = self._merge(cfg, self._load_session_overrides())
        cfg = self._inject_env_vars(cfg)

        log("Final unified config loaded.")
        return cfg
