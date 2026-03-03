from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import page_header, section_title


def render_risk_attribution(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    trades = context.backtest["trades"]
    if trades.empty:
        st.info("Risk attribution requires a saved trade ledger.")
        return

    page_header(
        "Risk Attribution",
        "Attribution built from the canonical trade ledger instead of stale candle-only assumptions.",
        kicker="Risk Lens",
    )

    by_regime = trades.groupby("regime", dropna=False).agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        avg_r=("r", "mean"),
    ).reset_index()
    by_session = trades.groupby("session", dropna=False).agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        avg_r=("r", "mean"),
    ).reset_index()
    by_leg = trades.groupby("leg", dropna=False).agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        avg_r=("r", "mean"),
    ).reset_index()

    section_title("PnL by Regime", "How the system performed in each attached regime state")
    st.altair_chart(
        alt.Chart(by_regime).mark_bar(color="#ffb000").encode(x="pnl:Q", y=alt.Y("regime:N", sort="-x"), tooltip=list(by_regime.columns)).properties(height=260),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        section_title("PnL by Session", "Execution-session contribution")
        st.dataframe(by_session, use_container_width=True, hide_index=True)
    with col2:
        section_title("PnL by Leg", "Core versus runner extraction")
        st.dataframe(by_leg, use_container_width=True, hide_index=True)

    stress = trades.copy()
    stress["hazard_bucket"] = pd.qcut(stress["hazard_entry"].fillna(0.0), q=min(4, max(len(stress), 1)), duplicates="drop")
    stress["evr_bucket"] = pd.qcut(stress["evr"].fillna(0.0), q=min(4, max(len(stress), 1)), duplicates="drop")

    section_title("Hazard / EVR Stress Grid", "PnL density across risk-at-entry and expected-value bins")
    pivot = stress.pivot_table(index="hazard_bucket", columns="evr_bucket", values="pnl", aggfunc="mean")
    st.dataframe(pivot, use_container_width=True)
