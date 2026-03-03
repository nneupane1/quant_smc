"""
Feature engineering package.

Use lazy exports so optional feature stacks such as regime/HDBSCAN are only
imported when requested explicitly.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "TFResampler": "quant_system.features.resampler",
    "EMAFeatureBuilder": "quant_system.features.ema_features",
    "LiquidityFeatureBuilder": "quant_system.features.liquidity_features",
    "VolatilityFeatureBuilder": "quant_system.features.volatility_features",
    "RegimeFeatureBuilder": "quant_system.features.regime_features",
    "FeatureStore": "quant_system.features.feature_store",
    "FeaturePreprocessor": "quant_system.features.preprocessing",
    "SwingHighLowDetector": "quant_system.features.smc.swings",
    "BOSCHOCHDetector": "quant_system.features.smc.bos_choch",
    "FVGDetector": "quant_system.features.smc.fvg",
    "LiquiditySweepDetector": "quant_system.features.smc.sweep",
    "OrderBlockDetector": "quant_system.features.smc.zones",
    "StructureContextBuilder": "quant_system.features.smc.structure_context",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
