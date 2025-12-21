"""
model_ensemble package
Provides:
 • ConsensusEngine      – multi-model probability fusion
 • WeightScheduler      – regime-conditioned ensemble weighting
 • DisagreementDetector – ensemble uncertainty metrics
 • ModelGovernor        – deployment, promotion, rollback, safety gating
 • ModelRegistry        – versioned artifact storage (already built earlier)

All components integrate with:
 • ForwardEngine (live & replay)
 • Backtester
 • Training pipelines
"""

from quant_system.model_ensemble.consensus_engine import ConsensusEngine
from quant_system.model_ensemble.weight_scheduler import WeightScheduler
from quant_system.model_ensemble.disagreement_detector import DisagreementDetector
from quant_system.model_ensemble.model_governor import ModelGovernor

__all__ = [
    "ConsensusEngine",
    "WeightScheduler",
    "DisagreementDetector",
    "ModelGovernor"
]
