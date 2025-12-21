"""
tv_chart.py
Python wrapper for the TradingView-style React chart component.

This file sends incremental candle updates to the React frontend, which
maintains full internal chart state (zoom, pan, overlays) and updates
without flicker or reinitialization.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
from streamlit.components.v1 import declare_component


# React bundle registration
_tv_chart_component = declare_component(
    "tv_chart_component",
    path=str(Path(__file__).parent.parent / "tv_chart_build")
)


def _prepare_candle_payload(state) -> dict:
    """
    Converts StateModel's candle history into a payload that React-based chart understands.
    """
    df = state.candles_1m.copy()
    if df.empty:
        return {"candles": []}

    df = df.sort_values("timestamp")

    return {
        "candles": [
            {
                "t": int(row.timestamp),
                "o": float(row.open),
                "h": float(row.high),
                "l": float(row.low),
                "c": float(row.close),
                "v": float(row.volume)
            }
            for row in df.itertuples()
        ]
    }


def tv_chart_component(state):
    """
    Renders the TradingView-style chart via the React component.
    """
    payload = _prepare_candle_payload(state)
    _tv_chart_component(key="tv_chart_component", data=payload)


def render_tv_chart(df: pd.DataFrame, key: str = "tv_chart_fallback"):
    """
    Fallback Altair candlestick chart for simple rendering when the React component is not wired.
    Expects columns: timestamp/dt, open, high, low, close.
    """
    if df.empty:
        st.info("No candle data to display.")
        return

    df = df.copy()
    if "dt" not in df.columns and "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df["color"] = df["close"] > df["open"]

    base = alt.Chart(df).encode(x="dt:T")
    rule = base.mark_rule().encode(
        y="low:Q",
        y2="high:Q",
        color=alt.condition("datum.close > datum.open", alt.value("#26a69a"), alt.value("#ef5350")),
    )
    bar = base.mark_bar().encode(
        y="open:Q",
        y2="close:Q",
        color=alt.condition("datum.close > datum.open", alt.value("#26a69a"), alt.value("#ef5350")),
    )
    st.altair_chart(rule + bar, use_container_width=True, theme="dark")
