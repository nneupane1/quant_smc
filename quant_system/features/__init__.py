"""
Feature Engineering Module
--------------------------

Provides:
- Timeframe-aware resampling utilities
- Full SMC (Smart Money Concepts) feature extractors
- EMA, volatility, liquidity, regime features
- FeatureStore for incremental storage and retrieval
- Preprocessing utilities (scaling, winsorization, joins)

All functions are deterministic, non-repainting, and aligned with
the 15m → 1h → 6h → 12h multi-timeframe pipeline.
"""

from .resampler import TFResampler
from .ema_features import EMAFeatureBuilder
from .liquidity_features import LiquidityFeatureBuilder
from .volatility_features import VolatilityFeatureBuilder
from .regime_features import RegimeFeatureBuilder
from .feature_store import FeatureStore
from .preprocessing import FeaturePreprocessor

# SMC submodules
from .smc.swings import SwingHighLowDetector
from .smc.bos_choch import BOSCHOCHDetector
from .smc.fvg import FVGDetector
from .smc.sweep import LiquiditySweepDetector
from .smc.zones import OrderBlockDetector
from .smc.structure_context import StructureContextBuilder
