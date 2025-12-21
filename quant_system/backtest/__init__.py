"""
Backtesting package: core engines, replay tools, and visuals/reporting.
"""

from quant_system.backtest.core.backtester import Backtester
from quant_system.backtest.core.execution_simulator import ExecutionSimulator, Position
from quant_system.backtest.core.metrics import BacktestMetrics
from quant_system.backtest.core.trade_log import TradeLog

from quant_system.backtest.visuals.report_builder import build_report, launch_dashboard

__all__ = [
    "Backtester",
    "ExecutionSimulator",
    "Position",
    "BacktestMetrics",
    "TradeLog",
    "build_report",
    "launch_dashboard",
]
