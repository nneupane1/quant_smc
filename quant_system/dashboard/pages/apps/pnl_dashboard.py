from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import (
    DashboardContext,
    build_equity_curve,
    summarize_trades,
)
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def _backtest_bundle(context: DashboardContext) -> dict:
    return context.backtest


def _equity_chart(equity_curve: pd.DataFrame) -> None:
    if equity_curve.empty:
        st.info("No equity curve available.")
        return
    curve = equity_curve.copy()
    if "timestamp" not in curve.columns:
        st.info("Equity curve is missing timestamps.")
        return
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
    chart = (
        alt.Chart(curve)
        .mark_line(color="#ffb000", strokeWidth=2.2)
        .encode(x="timestamp:T", y="equity:Q", tooltip=["timestamp:T", "equity:Q", "drawdown:Q"])
        .properties(height=310)
    )
    st.altair_chart(chart, use_container_width=True)


def _drawdown_chart(equity_curve: pd.DataFrame) -> None:
    if equity_curve.empty:
        st.info("No drawdown data available.")
        return
    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
    chart = (
        alt.Chart(curve)
        .mark_area(color="#ff6b6b", opacity=0.52)
        .encode(x="timestamp:T", y="drawdown:Q", tooltip=["timestamp:T", "drawdown:Q"])
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def _daily_monthly_panels(bundle: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        section_title("Daily PnL", "Grouped from canonical trade ledger")
        daily = bundle["daily"]
        if daily.empty:
            st.info("No daily report available.")
        else:
            st.dataframe(daily, use_container_width=True, hide_index=True)
    with col2:
        section_title("Monthly PnL", "Cycle-level view")
        monthly = bundle["monthly"]
        if monthly.empty:
            st.info("No monthly report available.")
        else:
            st.dataframe(monthly, use_container_width=True, hide_index=True)


def _scatter(trades: pd.DataFrame) -> None:
    if trades.empty:
        st.info("No closed trades available.")
        return
    chart = (
        alt.Chart(trades)
        .mark_circle(size=70, opacity=0.74)
        .encode(
            x=alt.X("evr:Q", title="EVR"),
            y=alt.Y("r:Q", title="Realized R"),
            color=alt.Color("result:N", scale=alt.Scale(domain=["win", "loss", "flat"], range=["#3ddc97", "#ff6b6b", "#6ea8fe"])),
            tooltip=["trade_id", "asset", "side", "entry_ts:T", "evr:Q", "r:Q", "pnl:Q", "tier"],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _tier_breakdown(trades: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        section_title("Tier Breakdown", "Ranking distribution")
        if trades.empty:
            st.info("No tier data available.")
        else:
            tier_df = trades.groupby("tier", dropna=False).agg(trades=("trade_id", "count"), pnl=("pnl", "sum")).reset_index()
            st.dataframe(tier_df, use_container_width=True, hide_index=True)
    with col2:
        section_title("Session Breakdown", "Execution session quality")
        if trades.empty:
            st.info("No session data available.")
        else:
            session_df = trades.groupby("session", dropna=False).agg(trades=("trade_id", "count"), pnl=("pnl", "sum"), avg_r=("r", "mean")).reset_index()
            st.dataframe(session_df, use_container_width=True, hide_index=True)


def render_home(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    backtest = _backtest_bundle(context)
    forward = context.forward
    summary = backtest["summary"] or summarize_trades(backtest["trades"])

    page_header(
        "Quant System Terminal",
        "12h regime, 6h structure, 1h flow ML, and 15m execution ML in one operator cockpit.",
        kicker="Mission Control",
    )
    st.markdown(
        f"{status_badge('12h HMM/HDBSCAN', 'good')} "
        f"{status_badge('1h Flow ML active', 'good')} "
        f"{status_badge('15m Execution stack', 'good')}",
        unsafe_allow_html=True,
    )
    metric_grid(
        [
            {"label": "Ending Equity", "value": f"${summary['ending_equity']:,.2f}"},
            {"label": "Total PnL", "value": f"${summary['total_pnl']:,.2f}", "delta": f"{summary['win_rate'] * 100:.1f}% WR"},
            {"label": "Active Model Version", "value": model_version},
            {"label": "Forward Equity", "value": f"${forward['state'].get('equity', 0.0):,.2f}"},
        ]
    )

    left, right = st.columns([1.7, 1.1])
    with left:
        section_title("Backtest Equity", "Current saved backtest baseline")
        _equity_chart(backtest["equity_curve"])
    with right:
        section_title("Forward Snapshot", "Latest adapter state")
        st.json(
            {
                "equity": forward["state"].get("equity"),
                "free_capital": forward["state"].get("free_capital"),
                "locked_profit": forward["state"].get("locked_profit"),
                "confluence": forward["state"].get("confluence"),
                "hazard": forward["state"].get("hazard"),
                "flow_1h": forward["state"].get("flow_1h"),
                "open_trades": len(forward["state"].get("open_trades", {})),
            }
        )

    _daily_monthly_panels(backtest)


def render_backtest(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    bundle = _backtest_bundle(context)
    trades = bundle["trades"]
    summary = bundle["summary"] or summarize_trades(trades)
    equity_curve = bundle["equity_curve"] if not bundle["equity_curve"].empty else build_equity_curve(
        trades, starting_equity=float(summary.get("starting_equity", 20_000.0))
    )

    page_header(
        "Backtest Results",
        "Canonical ledger, equity curve, and execution diagnostics generated from the repaired backtest loop.",
        kicker="Historical Diagnostics",
    )
    metric_grid(
        [
            {"label": "Trades", "value": f"{summary['trades']}"},
            {"label": "Win Rate", "value": f"{summary['win_rate'] * 100:.2f}%"},
            {"label": "Avg R", "value": f"{summary['avg_r']:.2f}"},
            {"label": "Max Drawdown", "value": f"{summary['max_drawdown']:,.2f}"},
        ]
    )

    col1, col2 = st.columns([1.9, 1.1])
    with col1:
        section_title("Equity Curve", "From saved backtest artifacts")
        _equity_chart(equity_curve)
    with col2:
        section_title("Drawdown", "Peak-to-trough capital stress")
        _drawdown_chart(equity_curve)

    _daily_monthly_panels(bundle)

    section_title("EVR vs Realized R", "Entry quality against delivered R-multiple")
    _scatter(trades)

    _tier_breakdown(trades)

    section_title("Trade Ledger", "Canonical schema used across dashboard and replay")
    st.dataframe(
        trades[
            [
                "trade_id",
                "entry_ts",
                "exit_ts",
                "asset",
                "side",
                "tier",
                "leg",
                "conf",
                "evr",
                "r",
                "pnl",
                "reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
