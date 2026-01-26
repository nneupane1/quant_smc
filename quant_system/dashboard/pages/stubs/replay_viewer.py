"""
Replay Viewer Dashboard
----------------------

This Streamlit page combines a timeline overview with a replay viewer for
historical backtests or forward tests.  Users can explore a heatmap of
model metrics over time, select a window of interest and inspect the
underlying price action and trades for that window.  This tool bridges
the gap between high‑level timelines and detailed replays.

The implementation is intentionally modular: functions are provided for
loading candles and trades, aggregating metrics over arbitrary time
periods, building Altair charts and rendering interactive controls.  The
page accepts query parameters to specify data sources and supports custom
aggregation intervals.  See inline comments for guidance on extending
the viewer (e.g. adding support for additional metrics or data sources).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


@dataclass
class Candle:
    """Represents a single OHLCV candle."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @classmethod
    def from_series(cls, s: pd.Series) -> "Candle":
        return cls(
            timestamp=pd.to_datetime(s["timestamp"]),
            open=float(s["open"]),
            high=float(s["high"]),
            low=float(s["low"]),
            close=float(s["close"]),
            volume=float(s["volume"]) if "volume" in s else None,
        )


def load_candles(path: Path) -> pd.DataFrame:
    """
    Load candle data from a CSV file.

    The CSV is expected to contain at least ``timestamp``, ``open``, ``high``,
    ``low`` and ``close`` columns.  Additional columns (e.g. volume,
    confluence, evr, hazard) are preserved.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Convert timestamp to datetime for proper Altair handling
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_trades(path: Path) -> pd.DataFrame:
    """
    Load trade events from a CSV file.

    The CSV should have at least ``entry_time``, ``exit_time``, ``side`` and
    ``pnl`` columns.  Timestamps are converted to datetime.
    """
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path)
    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    if "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    return trades


def aggregate_metric(df: pd.DataFrame, metric: str, interval: str) -> pd.DataFrame:
    """
    Aggregate a metric over a specified time interval.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``timestamp`` column and the metric to
        aggregate.
    metric : str
        Column name of the metric to average.
    interval : str
        Resampling interval (e.g. ``'4H'`` for 4‑hour bins, ``'1D'`` for
        daily, ``'1W'`` for weekly).  See Pandas offset aliases.

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with ``timestamp`` as the left edge of each
        bin and the mean metric value in that bin.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    resampled = (
        df.set_index("timestamp")[metric]
        .resample(interval)
        .mean()
        .dropna()
        .reset_index()
    )
    return resampled


def render_timeline(df: pd.DataFrame, metric: str, interval: str) -> None:
    """
    Render a heatmap timeline for a given metric.

    Parameters
    ----------
    df : pandas.DataFrame
        Candle DataFrame.
    metric : str
        Metric column to visualise (e.g. ``'confluence'``, ``'evr'`` or
        ``'hazard'``).  Values are normalised between 0 and 1 across the
        dataset.
    interval : str
        Resampling interval for the timeline.
    """
    agg = aggregate_metric(df, metric, interval)
    if agg.empty:
        st.info(f"No data available for metric '{metric}'.")
        return
    # Normalise values for colour mapping
    values = agg[metric]
    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-9)
    heat_df = pd.DataFrame({"timestamp": agg["timestamp"], "value": norm_values})
    heat_chart = (
        alt.Chart(heat_df)
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(title="Time")),
            y=alt.Y("value:Q", axis=alt.Axis(title="Normalised Value")),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="turbo"),
                legend=None,
            ),
            tooltip=["timestamp:T", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=100)
    )
    st.altair_chart(heat_chart, use_container_width=True)


def render_segment(df: pd.DataFrame, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Render price action and trades for a selected time window.

    This function displays an OHLC chart with vertical lines marking trade
    entries and exits.  It also shows a table of trades executed during
    the selected window.
    """
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    segment = df.loc[mask]
    if segment.empty:
        st.info("No data in selected window.")
        return
    # Candlestick chart using Altair.  Altair does not have a built‑in
    # candlestick mark, so we compose high/low and open/close segments.
    base = alt.Chart(segment).encode(x="timestamp:T")
    rule = base.mark_rule().encode(y="low:Q", y2="high:Q")
    bar = base.mark_bar().encode(
        y="open:Q",
        y2="close:Q",
        color=alt.condition("close > open", alt.value("#00C853"), alt.value("#D50000")),
    )
    price_chart = (rule + bar).properties(height=300, title="Price Action")
    st.altair_chart(price_chart, use_container_width=True)
    # Display trades in window
    trade_mask = (trades["entry_time"] >= start) & (trades["entry_time"] <= end)
    trades_window = trades.loc[trade_mask]
    if not trades_window.empty:
        st.subheader("Trades in Window")
        st.dataframe(trades_window, use_container_width=True)
    else:
        st.info("No trades executed in this window.")


def render_replay_viewer() -> None:
    """
    Main entrypoint for the replay viewer Streamlit page.

    Users can choose aggregation intervals and metrics for the timeline, then
    select a window to inspect using a slider.  Charts update dynamically
    based on the selected metric and window.
    """
    st.set_page_config(page_title="Replay Viewer", layout="wide")
    st.title("Replay Viewer")
    params = st.experimental_get_query_params()
    candle_file = Path(params.get("candle_file", ["backtest_outputs/candles.csv"])[0])
    trades_file = Path(params.get("trades_file", ["backtest_outputs/trades.csv"])[0])
    candles = load_candles(candle_file)
    trades = load_trades(trades_file)
    if candles.empty:
        st.warning("No candle data found. Please provide a valid CSV via `?candle_file=...`. ")
        return
    # Metric selection
    metric_options = [c for c in ["confluence", "evr", "hazard"] if c in candles.columns]
    if not metric_options:
        metric_options = ["close"]
    metric = st.sidebar.selectbox("Metric for Timeline", metric_options, index=0)
    interval = st.sidebar.selectbox(
        "Aggregation Interval",
        options=[("4H", "4 Hours"), ("1D", "1 Day"), ("1W", "1 Week")],
        format_func=lambda x: x[1],
        index=1,
    )[0]
    st.sidebar.markdown("---")
    st.sidebar.markdown("Select the time window to inspect:")
    min_time = candles["timestamp"].min()
    max_time = candles["timestamp"].max()
    start, end = st.sidebar.slider(
        "Window", min_value=min_time, max_value=max_time, value=(min_time, max_time), format="%Y-%m-%d"
    )
    # Render timeline heatmap
    st.subheader("Timeline")
    render_timeline(candles, metric, interval)
    st.subheader("Detailed View")
    render_segment(candles, trades, start, end)


if __name__ == "__main__":
    render_replay_viewer()
