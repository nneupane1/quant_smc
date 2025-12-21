"""
Dashboard components package.
Python components live in quant_system.dashboard.components.py.
JS/CSS assets live under quant_system.dashboard.components.js.
"""

from quant_system.dashboard.components.py.asset_selector import AssetSelector
from quant_system.dashboard.components.py.replay_control_bar import ReplayControlBar
from quant_system.dashboard.components.py.report_generator import ReportGenerator
from quant_system.dashboard.components.py.report_page import ReportPage
from quant_system.dashboard.components.py.trade_summary_panel import TradeSummaryPanel

__all__ = [
    "AssetSelector",
    "ReplayControlBar",
    "ReportGenerator",
    "ReportPage",
    "TradeSummaryPanel",
]
