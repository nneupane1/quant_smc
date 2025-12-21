"""
Regime modeling package exports.

Exposes the HMM and HDBSCAN trainers/configs for easy imports in CLIs.
"""

from quant_system.models.regime.hmm_trainer import HMMRegimeTrainer, HMMConfig

# Optional: expose HDBSCAN components if needed elsewhere
try:
    from quant_system.models.regime.hdbscan_trainer import (
        HDBSCANLiquidityTrainer,
        HDBSCANConfig,
    )
except Exception:  # pragma: no cover - hdbscan may be optional dependency
    HDBSCANLiquidityTrainer = None  # type: ignore
    HDBSCANConfig = None  # type: ignore

__all__ = [
    "HMMRegimeTrainer",
    "HMMConfig",
    "HDBSCANLiquidityTrainer",
    "HDBSCANConfig",
]
