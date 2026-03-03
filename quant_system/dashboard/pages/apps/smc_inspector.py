from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import page_header, section_title


def render_smc_inspector(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    candles = context.backtest["candles"]
    smc = context.backtest["smc_features"]
    if candles.empty:
        st.info("SMC inspector requires backtest candles.")
        return

    page_header(
        "SMC Inspector",
        "Inspect the engineered 15m execution frame and attached SMC context at any replayable point.",
        kicker="Structure Lens",
    )

    candle_df = candles.copy()
    ts_col = "timestamp" if "timestamp" in candle_df.columns else "dt"
    candle_df["timestamp"] = pd.to_datetime(candle_df[ts_col], errors="coerce")
    max_idx = len(candle_df) - 1
    idx = st.slider("Candle Index", min_value=0, max_value=max_idx, value=max_idx // 2 if max_idx > 1 else 0, step=1)
    window = candle_df.iloc[max(0, idx - 120): idx + 1].copy()

    render_tv_chart(window, key="smc_window")

    row = candle_df.iloc[idx].to_dict()
    smc_row = smc.iloc[idx].to_dict() if not smc.empty and idx < len(smc) else {}

    left, right = st.columns(2)
    with left:
        section_title("Market Row", "Execution-frame candle and engineered row")
        st.json(row)
    with right:
        section_title("SMC Row", "Attached structure features for the selected index")
        st.json(smc_row)
