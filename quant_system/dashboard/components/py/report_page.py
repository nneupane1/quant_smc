from __future__ import annotations

from pathlib import Path

import streamlit as st

from quant_system.backtest.report_generator import render_backtest_report
from quant_system.dashboard.data_access import build_context


def render_report_page(theme_choice: str = "bloomberg") -> None:
    context = build_context(theme_choice)
    trades = context.backtest["trades"]
    equity_curve = context.backtest["equity_curve"]
    if trades.empty:
        st.info("No backtest artifacts available.")
        return

    result = render_backtest_report(
        trades,
        equity_df=equity_curve if not equity_curve.empty else None,
        starting_equity=float(context.backtest["summary"].get("starting_equity", 20_000.0)),
        registry_dir=str(context.model_dir),
    )
    st.write(result["summary"])
    report_dir = Path(context.backtest_dir)
    st.caption(f"Artifacts: {report_dir}")


if __name__ == "__main__":
    render_report_page()
