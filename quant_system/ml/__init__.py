"""
ML package: training, prediction, and registry utilities.

Keep package imports lightweight so callers can import specific submodules
without triggering optional training dependencies up front.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "FeatureBuilder": "quant_system.ml.training.feature_builder",
    "LabelLoader": "quant_system.ml.training.label_loader",
    "ModelTrainer": "quant_system.ml.training.model_trainer",
    "ModelOptimizer": "quant_system.ml.training.model_optimizer",
    "ModelPredictor": "quant_system.ml.predict.model_predictor",
    "EmpiricalCalibrator": "quant_system.ml.predict.empirical_calibrator",
    "ModelRegistry": "quant_system.ml.registry.model_registry",
    "ModelVersioning": "quant_system.ml.registry.versioning",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
