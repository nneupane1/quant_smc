from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import metric_grid, page_header, section_title


def _filter_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    sides = st.sidebar.multiselect("Side", sorted(trades["side"].dropna().unique()), default=sorted(trades["side"].dropna().unique()))
    tiers = st.sidebar.multiselect("Tier", sorted(trades["tier"].dropna().unique()), default=sorted(trades["tier"].dropna().unique()))
    sessions = st.sidebar.multiselect("Session", sorted(trades["session"].dropna().unique()), default=sorted(trades["session"].dropna().unique()))
    regimes = st.sidebar.multiselect("Regime", sorted(trades["regime"].dropna().unique()), default=sorted(trades["regime"].dropna().unique()))
    min_r = st.sidebar.slider("Min R", min_value=float(trades["r"].min()), max_value=float(max(trades["r"].max(), 1.0)), value=float(min(trades["r"].min(), 0.0)))
    moonshots = st.sidebar.checkbox("Moonshots >= 5R", value=False)

    df = trades.copy()
    mask = (
        df["side"].isin(sides)
        & df["tier"].isin(tiers)
        & df["session"].isin(sessions)
        & df["regime"].isin(regimes)
        & (df["r"] >= min_r)
    )
    df = df.loc[mask]
    if moonshots:
        df = df[df["r"] >= 5.0]
    return df


def render_journal(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    trades = context.backtest["trades"]
    reasoning = context.backtest["reasoning"]

    page_header(
        "Trade Journal",
        "Review the canonical backtest ledger with filters, reasoning payloads, and replay-ready timestamps.",
        kicker="Post Trade Review",
    )

    if trades.empty:
        st.info("No trades found in the active backtest directory.")
        return

    filtered = _filter_trades(trades)
    metric_grid(
        [
            {"label": "Filtered Trades", "value": f"{len(filtered)}"},
            {"label": "Win Rate", "value": f"{(filtered['pnl'] > 0).mean() * 100:.2f}%"},
            {"label": "Avg R", "value": f"{filtered['r'].mean():.2f}"},
            {"label": "PnL", "value": f"${filtered['pnl'].sum():,.2f}"},
        ]
    )

    section_title("Trade Ledger", "Filtered trade journal view")
    st.dataframe(
        filtered[
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
                "session",
                "regime",
                "reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    trade_ids = filtered["trade_id"].tolist()
    if not trade_ids:
        st.info("No trades match the active filters.")
        return

    selected_trade_id = st.selectbox("Inspect Trade", trade_ids)
    trade = filtered[filtered["trade_id"] == selected_trade_id].iloc[0]
    detail_col, reason_col = st.columns([1.2, 1.1])

    with detail_col:
        section_title("Trade Detail", "Execution snapshot")
        st.json(
            {
                "trade_id": trade["trade_id"],
                "asset": trade["asset"],
                "side": trade["side"],
                "tier": trade["tier"],
                "leg": trade["leg"],
                "entry_ts": str(trade["entry_ts"]),
                "exit_ts": str(trade["exit_ts"]),
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "stop_price": trade["stop_price"],
                "conf": trade["conf"],
                "evr": trade["evr"],
                "r": trade["r"],
                "pnl": trade["pnl"],
                "reason": trade["reason"],
            }
        )

    with reason_col:
        section_title("Reasoning Payload", "If recorded during backtest/forward execution")
        payload = reasoning.get(str(selected_trade_id), {})
        if payload:
            st.json(payload)
        else:
            st.info("No reasoning payload stored for this trade.")

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Journal", csv_data, file_name="trade_journal.csv", mime="text/csv")
