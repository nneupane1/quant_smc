"""
Compatibility wrapper for the authoritative replay controller.

The dashboard JS replay widgets should use the same controller implementation as
the repaired backtest package, not a stale copy with old execution imports.
"""

from quant_system.backtest.replay.replay_controller import ReplayController, ReplayState

__all__ = ["ReplayController", "ReplayState"]
