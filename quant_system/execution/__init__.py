"""
Execution package: gating, risk, and adapters.
"""

from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.gating.profit_ladder import ProfitLadderManager
from quant_system.execution.gating.tiering import TieringEngine

from quant_system.execution.risk.capital_allocator import CapitalAllocator
from quant_system.execution.risk.compound_cooling import CompoundCoolingPolicy
from quant_system.execution.risk.mpc_risk import MPCRiskManager
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.execution.risk.exposure_tracker import ExposureTracker
from quant_system.execution.risk.cooling_engine import CoolingEngine

from quant_system.execution.adapters.order_adapter import OrderAdapter

__all__ = [
    "ConfluenceEngine",
    "EVRCalculator",
    "GateEvaluator",
    "HazardTrailingEngine",
    "ProfitLadderManager",
    "TieringEngine",
    "CapitalAllocator",
    "CompoundCoolingPolicy",
    "MPCRiskManager",
    "PositionSizer",
    "ExposureTracker",
    "CoolingEngine",
    "OrderAdapter",
]
