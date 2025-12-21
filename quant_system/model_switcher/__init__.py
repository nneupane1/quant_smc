"""
model_switcher package
Provides regime-conditioned exclusive model selection:

 • ModelSelector       – main switching controller
 • RegimeRules         – rule engine mapping regime → preferred model
 • PerformanceMemory   – stores historical performance per model version

This module allows switching between specialist model versions
based on HMM regimes, liquidity states, volatility buckets, and past performance.
"""

from quant_system.model_switcher.selector import ModelSelector
from quant_system.model_switcher.regime_rules import RegimeRules
from quant_system.model_switcher.performance_memory import PerformanceMemory

__all__ = [
    "ModelSelector",
    "RegimeRules",
    "PerformanceMemory"
]
