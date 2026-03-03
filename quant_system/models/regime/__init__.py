"""
Regime-model package exports.
"""

try:
    from .hmm_regime import HMMRegimeConfig, HMMRegimeTrainer
except Exception:  # pragma: no cover - optional dependency
    HMMRegimeConfig = None  # type: ignore
    HMMRegimeTrainer = None  # type: ignore

try:
    from .hmm_trainer import HMMConfig, HMMTrainer
except Exception:  # pragma: no cover - optional dependency
    HMMConfig = None  # type: ignore
    HMMTrainer = None  # type: ignore

try:
    from .hdbscan_trainer import HDBSCANConfig, HDBSCANTrainer
except Exception:  # pragma: no cover - optional dependency
    HDBSCANConfig = None  # type: ignore
    HDBSCANTrainer = None  # type: ignore

__all__ = [
    "HMMConfig",
    "HMMTrainer",
    "HMMRegimeConfig",
    "HMMRegimeTrainer",
    "HDBSCANConfig",
    "HDBSCANTrainer",
]
