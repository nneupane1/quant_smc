from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.backtest.replay_controller import ReplayController
from quant_system.dashboard.data_access import DashboardContext, build_equity_curve
from quant_system.dashboard.ui import metric_grid, page_header, section_title
from quant_system.ml.registry.model_registry import ModelRegistry


def _replay_controller(context: DashboardContext):
    try:
        registry = ModelRegistry(str(context.model_dir)) if context.model_dir.exists() else None
    except Exception:
        registry = None
    return ReplayController(
        candles_15m=context.backtest["candles"],
        smc_features=context.backtest["smc_features"],
        execution_log=context.backtest["execution_log"] if not context.backtest["execution_log"].empty else context.backtest["trades"],
        model_bundle=registry,
        config=context.config,
    )


def render_research_audit(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    trades = context.backtest["trades"]
    summary = context.backtest["summary"]

    page_header(
        "Research & Audit",
        "Backtest, replay, ledger, and reasoning reconstruction surfaces for full operator auditability.",
        kicker="Audit Trail",
    )
    metric_grid(
        [
            {"label": "Trades", "value": f"{int(summary.get('trades') or 0)}"},
            {"label": "Ending Equity", "value": f"${float(summary.get('ending_equity') or 0.0):,.2f}"},
            {"label": "Win Rate", "value": f"{float(summary.get('win_rate') or 0.0) * 100:.2f}%"},
            {"label": "Backtest PnL", "value": f"${float(summary.get('total_pnl') or 0.0):,.2f}"},
        ]
    )

    tabs = st.tabs(["Backtest", "Replay Snapshot", "Ledger", "Reasoning"])
    with tabs[0]:
        section_title("Backtest Equity", "Historical performance baseline")
        equity_curve = context.backtest["equity_curve"]
        if equity_curve.empty:
            equity_curve = build_equity_curve(trades)
        if equity_curve.empty:
            st.info("No equity curve available.")
        else:
            curve = equity_curve.copy()
            curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
            chart = (
                alt.Chart(curve)
                .mark_line(color="#6ea8fe", strokeWidth=2.1)
                .encode(x="timestamp:T", y="equity:Q", tooltip=["timestamp:T", "equity:Q"])
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        st.dataframe(context.backtest["daily"], use_container_width=True, hide_index=True) if not context.backtest["daily"].empty else None
    with tabs[1]:
        section_title("Replay Snapshot", "Current payload at the end of the historical tape")
        if context.backtest["candles"].empty:
            st.info("No replay candles available.")
        else:
            controller = _replay_controller(context)
            controller.jump_to(max(len(context.backtest["candles"]) - 1, 0))
            st.json(controller.render_payload())
    with tabs[2]:
        section_title("Trade Ledger", "Canonical audit ledger")
        st.dataframe(trades, use_container_width=True, hide_index=True) if not trades.empty else st.info("No trades available.")
    with tabs[3]:
        section_title("Reasoning Store", "Persisted reasoning payloads if available")
        reasoning = context.backtest["reasoning"]
        if reasoning:
            preview = dict(list(reasoning.items())[:10])
            st.json(preview)
        else:
            st.info("No persisted reasoning store found.")
