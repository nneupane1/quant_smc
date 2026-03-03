from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext, normalize_trade_frame
from quant_system.dashboard.ui import metric_grid, page_header, section_title


def _forward_closed_trades(context: DashboardContext) -> pd.DataFrame:
    rows = []
    for event in context.forward["events"]:
        if event.get("event_type") not in {"exit", "exit_trade", "closed_trade"}:
            continue
        payload = event.get("payload", {}) or {}
        rows.append(
            {
                "trade_id": event.get("trade_id"),
                "asset": payload.get("asset"),
                "side": payload.get("side"),
                "entry_ts": payload.get("entry_ts") or payload.get("opened_at") or event.get("timestamp"),
                "exit_ts": payload.get("exit_ts") or event.get("timestamp"),
                "entry_price": payload.get("entry_price"),
                "exit_price": payload.get("exit_price"),
                "pnl": payload.get("pnl", 0.0),
                "r": payload.get("r", payload.get("r_mult", 0.0)),
                "tier": payload.get("tier"),
                "conf": payload.get("conf"),
                "evr": payload.get("evr"),
                "reason": payload.get("reason", event.get("event_type")),
                "leg": payload.get("leg"),
            }
        )
    return normalize_trade_frame(pd.DataFrame(rows))


def render_trade_log(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    backtest_trades = context.backtest["trades"]
    forward_trades = _forward_closed_trades(context)
    all_trades = normalize_trade_frame(pd.concat([backtest_trades, forward_trades], ignore_index=True))

    page_header(
        "Trade Log",
        "Unified trade tape across persisted backtests and in-memory forward/live exits.",
        kicker="Ledger",
    )
    if all_trades.empty:
        st.info("No closed trades available yet.")
        return

    metric_grid(
        [
            {"label": "Trades", "value": f"{len(all_trades)}"},
            {"label": "Win Rate", "value": f"{(all_trades['pnl'] > 0).mean() * 100:.2f}%"},
            {"label": "Avg R", "value": f"{all_trades['r'].mean():.2f}"},
            {"label": "Expectancy", "value": f"${all_trades['pnl'].mean():,.2f}"},
        ]
    )

    equity_curve = all_trades.sort_values("entry_ts")[["entry_ts", "pnl"]].copy()
    equity_curve["equity"] = 20_000 + equity_curve["pnl"].fillna(0.0).cumsum()
    section_title("Equity Trace", "Unified ledger progression")
    chart = (
        alt.Chart(equity_curve)
        .mark_line(color="#3ddc97", strokeWidth=2.2)
        .encode(x="entry_ts:T", y="equity:Q", tooltip=["entry_ts:T", "equity:Q"])
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

    section_title("Trade Table", "Backtest and forward exits in one frame")
    st.dataframe(
        all_trades[
            [
                "trade_id",
                "entry_ts",
                "exit_ts",
                "asset",
                "side",
                "tier",
                "leg",
                "r",
                "pnl",
                "reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    daily = all_trades.assign(date=all_trades["entry_ts"].dt.date).groupby("date").agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        avg_r=("r", "mean"),
    ).reset_index()
    monthly = all_trades.assign(month=all_trades["entry_ts"].dt.to_period("M").astype(str)).groupby("month").agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        avg_r=("r", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        section_title("Daily", "Operator review cadence")
        st.dataframe(daily, use_container_width=True, hide_index=True)
    with col2:
        section_title("Monthly", "Longer-horizon performance drift")
        st.dataframe(monthly, use_container_width=True, hide_index=True)
