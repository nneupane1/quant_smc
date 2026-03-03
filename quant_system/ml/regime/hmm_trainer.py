"""
Compatibility wrapper over the canonical higher-timeframe regime HMM trainer.

The authoritative implementation lives in
`quant_system.ml.training.regime_hmm_trainer` and is intended for 6h/12h
regime-state training. This module preserves the older import surface.
"""

from quant_system.ml.training.regime_hmm_trainer import (
    RegimeHMMConfig as HMMConfig,
    RegimeHMMTrainer,
)

__all__ = ["HMMConfig", "RegimeHMMTrainer"]
