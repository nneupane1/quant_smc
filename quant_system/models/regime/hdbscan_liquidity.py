"""
Compatibility wrapper for HDBSCAN liquidity clustering.
"""

from quant_system.models.regime.hdbscan_trainer import (
    HDBSCANConfig,
    HDBSCANTrainer as HDBSCANLiquidity,
)

__all__ = ["HDBSCANConfig", "HDBSCANLiquidity"]
