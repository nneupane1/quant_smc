"""
Simple monotonic model versioning system.
"""

import os
import json
from typing import Dict
from quant_system.utils.logger import log


class ModelVersionManager:
    """
    Manages sequential model versions stored in a JSON index.
    """

    def __init__(self, index_path: str = ".model_versions.json"):
        self.index_path = index_path
        self._versions = self._load()
        log(f"ModelVersionManager using index: {self.index_path}")

    def _load(self) -> Dict[str, int]:
        if not os.path.exists(self.index_path):
            return {}
        with open(self.index_path, "r") as f:
            try:
                return json.load(f)
            except:
                return {}

    def _save(self):
        with open(self.index_path, "w") as f:
            json.dump(self._versions, f, indent=2)

    def new_version(self, model_name: str) -> str:
        """
        Increment version for model_name and save.
        """
        current = self._versions.get(model_name, 0)
        new = current + 1
        self._versions[model_name] = new
        self._save()
        version_str = f"v{new:04d}"

        log(f"New version for {model_name}: {version_str}")
        return version_str

    def latest(self, model_name: str) -> str:
        """
        Return latest version string for model_name.
        """
        if model_name not in self._versions:
            raise ValueError(f"No version found for model: {model_name}")
        v = self._versions[model_name]
        return f"v{v:04d}"


# Backward-compatible alias
ModelVersioning = ModelVersionManager
