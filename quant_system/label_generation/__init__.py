"""
Label Generation Package
------------------------

This package produces all supervised learning labels used by the system’s
specialist models, meta-models, hazard model, and quantile forecaster.

Labels follow strict multi-timeframe, no-repaint, closed-bar rules and align to
the canonical execution TF (15m). This module is consumed by:

    - Liquidity-Flow Model (12 × 15m horizon)
    - BOS Continuation Model (48 × 15m horizon)
    - Micro-Momentum Model (4–8 × 15m horizon)
    - EOP – Expected Opportunity Probability (96 × 15m)
    - EDP – Expected Drawdown Probability  (96 × 15m)
    - Hazard (survival) Model (time-to-failure, 48 × 15m)
    - Meta-model gating / Confluence scoring
    - Backtester reliability checks

Label Philosophy:
    1. No future data leakage — labels rely only on fully closed bars.
    2. Multi-timeframe correctness — BOS/CHOCH from 1h/6h/12h; sweeps from
       15m/1h; EVR-based success criteria respect stop/target envelopes.
    3. R-based outcomes — success defined in risk units, not raw prices.
    4. Timeboxing — each label has a fixed horizon to reflect tradeable windows.
    5. Reproducibility — deterministic, timestamp-aligned outputs.

Modules:
    bos_continuation.py  → BOS continuation labels
    liquidity_flow.py    → sweep → displacement → continuation labels
    micro_momentum.py    → acceleration/exhaustion labels
    eop.py               → Expected Opportunity Probability labels
    edp.py               → Expected Drawdown Probability labels
    hazard.py            → survival time-to-failure labels
    utils.py             → shared utilities for label generation

All generators expose:
    generate_labels(candles, features, smc_structures, ...) → {ts → label}

This package is strictly side-effect-free and produces pure label dictionaries
for downstream preprocessing, training, and backtesting.
"""

from quant_system.label_generation.bos_continuation import BOSContinuationLabeler
from quant_system.label_generation.liquidity_flow import LiquidityFlowLabeler
from quant_system.label_generation.micro_momentum import MicroMomentumLabeler
from quant_system.label_generation.eop import EOPLabeler
from quant_system.label_generation.edp import EDPLabeler
from quant_system.label_generation.hazard import HazardLabeler
from quant_system.label_generation.label_builder import LabelBuilder

__all__ = [
    "BOSContinuationLabeler",
    "LiquidityFlowLabeler",
    "MicroMomentumLabeler",
    "EOPLabeler",
    "EDPLabeler",
    "HazardLabeler",
    "LabelBuilder",
]
