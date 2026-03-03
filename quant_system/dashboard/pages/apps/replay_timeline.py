from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import page_header, section_title


def render_replay_timeline(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    candles = context.backtest["candles"].copy()
    execution_log = context.backtest["execution_log"].copy()
    if candles.empty:
        st.info("Replay timeline requires saved candles in the active backtest directory.")
        return

    ts_col = "timestamp" if "timestamp" in candles.columns else "dt"
    candles["timestamp"] = pd.to_datetime(candles[ts_col], errors="coerce")
    metric = st.selectbox(
        "Heatmap Metric",
        [col for col in ["confluence", "conf", "evr", "hazard", "close"] if col in candles.columns] or ["close"],
        index=0,
    )
    interval = st.selectbox("Interval", ["15min", "1H", "4H", "1D"], index=2)

    page_header(
        "Replay Timeline",
        "Timeline heatmap for quickly locating high-confluence or high-risk zones before drilling into replay.",
        kicker="Navigator",
    )

    grouped = candles.set_index("timestamp")[[metric]].resample(interval).mean().dropna().reset_index()
    if grouped.empty:
        st.info("No timeline data available for the selected metric.")
        return

    section_title("Timeline Heatmap", "Resampled metric intensity")
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x="timestamp:T",
            y=alt.value(40),
            color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="turbo")),
            tooltip=["timestamp:T", f"{metric}:Q"],
        )
        .properties(height=120)
    )
    st.altair_chart(chart, use_container_width=True)

    section_title("Trade Events", "Replay anchor points")
    if execution_log.empty:
        st.info("No execution log found.")
    else:
        st.dataframe(execution_log.tail(200), use_container_width=True, hide_index=True)
