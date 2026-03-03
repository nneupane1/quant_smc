"""
ML regime helpers and trainers.
"""

try:
    from .hmm_trainer import HMMConfig, RegimeHMMTrainer
except Exception:  # pragma: no cover - optional dependency
    HMMConfig = None  # type: ignore
    RegimeHMMTrainer = None  # type: ignore

try:
    from .hdbscan_clustering import HDBSCANClusterer, HDBSCANConfig as ClusterHDBSCANConfig
except Exception:  # pragma: no cover - optional dependency
    HDBSCANClusterer = None  # type: ignore
    ClusterHDBSCANConfig = None  # type: ignore

try:
    from .hdbscan_trainer import HDBSCANConfig, LiquidityClusterTrainer
except Exception:  # pragma: no cover - optional dependency
    HDBSCANConfig = None  # type: ignore
    LiquidityClusterTrainer = None  # type: ignore

__all__ = [
    "HMMConfig",
    "RegimeHMMTrainer",
    "HDBSCANConfig",
    "LiquidityClusterTrainer",
    "ClusterHDBSCANConfig",
    "HDBSCANClusterer",
]
