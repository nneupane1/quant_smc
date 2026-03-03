"""
Compatibility wrapper over the canonical higher-timeframe HMM regime trainer.
"""

from quant_system.ml.training.regime_hmm_trainer import (
    RegimeHMMConfig as HMMRegimeConfig,
    RegimeHMMTrainer as HMMRegimeTrainer,
)

__all__ = ["HMMRegimeConfig", "HMMRegimeTrainer"]
