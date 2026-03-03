"""
Backtesting package: core engines, replay tools, and visuals/reporting.

Keep imports lazy so engine code does not require visualization dependencies
unless those reporting helpers are requested explicitly.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Backtester": "quant_system.backtest.core.backtester",
    "ExecutionSimulator": "quant_system.backtest.core.execution_simulator",
    "Position": "quant_system.backtest.core.execution_simulator",
    "BacktestMetrics": "quant_system.backtest.core.metrics",
    "TradeLog": "quant_system.backtest.core.trade_log",
    "ReplayController": "quant_system.backtest.replay.replay_controller",
    "build_report": "quant_system.backtest.visuals.report_builder",
    "launch_dashboard": "quant_system.backtest.visuals.report_builder",
    "generate_backtest_artifacts": "quant_system.backtest.report_generator",
    "render_backtest_report": "quant_system.backtest.report_generator",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
