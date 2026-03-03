from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext, build_equity_curve
from quant_system.dashboard.intelligence import operator_summary
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def _open_trades_frame(open_trades) -> pd.DataFrame:
    rows = []
    if isinstance(open_trades, dict):
        items = open_trades.items()
    elif isinstance(open_trades, list):
        items = enumerate(open_trades)
    else:
        items = []
    for trade_id, trade in items:
        if isinstance(trade, dict):
            rows.append(
                {
                    "trade_id": trade.get("trade_id", trade_id),
                    "asset": trade.get("asset"),
                    "side": trade.get("side"),
                    "leg": trade.get("leg") or (trade.get("metadata", {}) or {}).get("leg"),
                    "entry_price": trade.get("entry_price") or trade.get("entry"),
                    "stop_price": trade.get("stop_price") or trade.get("stop"),
                    "size_usd": trade.get("size_usd") or trade.get("size"),
                    "highest_r": trade.get("highest_r"),
                }
            )
    return pd.DataFrame(rows)


def render_mission_control(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    state = context.forward["state"]
    events = context.forward["events"]
    candles = context.forward["candles"] if isinstance(context.forward["candles"], pd.DataFrame) and not context.forward["candles"].empty else context.backtest["candles"]
    summary = operator_summary(context)
    readiness_badge = status_badge(
        "Cooling active" if state.get("cooling_to") else "Ready to deploy",
        "warn" if state.get("cooling_to") else "good",
    )
    regime_badge = status_badge(f"Regime {summary.get('regime_state') or 'unknown'}", "neutral")

    page_header(
        "Mission Control",
        "Live operator desk for account state, active trades, execution tape, and capital-cycle supervision.",
        kicker="Terminal",
    )
    st.markdown(
        f"{status_badge('Flow ML active', 'good')} "
        f"{regime_badge} "
        f"{readiness_badge}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Live Equity", "value": f"${float(summary.get('live_equity') or 0.0):,.2f}"},
            {"label": "Locked Profit", "value": f"${float(summary.get('locked_profit') or 0.0):,.2f}"},
            {"label": "Open Trades", "value": f"{int(summary.get('open_trades') or 0)}"},
            {"label": "1h Flow ML", "value": f"{float(summary.get('flow_1h') or 0.0):.3f}"},
            {"label": "Confluence", "value": f"{float(summary.get('confluence') or 0.0):.3f}"},
            {"label": "Hazard", "value": f"{float(summary.get('hazard') or 0.0):.3f}"},
        ]
    )

    chart_col, side_col = st.columns([2.1, 1.05])
    with chart_col:
        section_title("Execution Surface", "Primary market view with overlay-capable chart shell")
        render_tv_chart(candles if isinstance(candles, pd.DataFrame) else pd.DataFrame(), key="mission_control_chart")
    with side_col:
        section_title("Operator State", "Capital and deployment readiness")
        st.json(
            {
                "equity": state.get("equity"),
                "free_capital": state.get("free_capital"),
                "locked_profit": state.get("locked_profit"),
                "cooling_to": state.get("cooling_to"),
                "risk_mode": state.get("risk_mode"),
                "hedge_ratio": state.get("hedge_ratio"),
            }
        )

    tabs = st.tabs(["Command Deck", "Execution Tape", "Capital Trace"])
    with tabs[0]:
        section_title("Open Trades", "Core and runner inventory")
        open_trades = _open_trades_frame(state.get("open_trades", {}))
        if open_trades.empty:
            st.info("No open trades.")
        else:
            st.dataframe(open_trades, use_container_width=True, hide_index=True)
    with tabs[1]:
        section_title("Recent Events", "Forward/live event stream")
        if events:
            st.dataframe(pd.DataFrame(events[-120:]), use_container_width=True, hide_index=True)
        else:
            st.info("No events yet.")
    with tabs[2]:
        section_title("Backtest Baseline Equity", "Historical benchmark versus live capital cycle")
        equity_curve = context.backtest["equity_curve"]
        if equity_curve.empty:
            equity_curve = build_equity_curve(context.backtest["trades"])
        if equity_curve.empty:
            st.info("No equity curve available.")
        else:
            curve = equity_curve.copy()
            curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
            chart = (
                alt.Chart(curve)
                .mark_line(color="#ffb000", strokeWidth=2.2)
                .encode(x="timestamp:T", y="equity:Q", tooltip=["timestamp:T", "equity:Q"])
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
