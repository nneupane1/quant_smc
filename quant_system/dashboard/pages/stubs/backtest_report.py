from __future__ import annotations

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.pages.apps.pnl_dashboard import render_backtest


def render_backtest_report(theme_choice: str = "bloomberg") -> None:
    context = build_context(theme_choice)
    render_backtest(theme_choice, context.model_version, context=context)


if __name__ == "__main__":
    render_backtest_report()
