"""
Compatibility wrapper for legacy clustering imports.

Delegates to the canonical `quant_system.ml.regime.hdbscan_clustering`
implementation.
"""

from quant_system.ml.regime.hdbscan_clustering import (
    HDBSCANConfig,
    HDBSCANClusterer,
)

__all__ = ["HDBSCANConfig", "HDBSCANClusterer"]
