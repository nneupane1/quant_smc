from typing import Optional

import pandas as pd


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def _optional_altair():
    try:
        import altair as alt

        return alt
    except Exception:
        return None


def _heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, render: bool = True):
    if df.empty:
        return df
    if not render:
        return df

    st = _optional_streamlit()
    alt = _optional_altair()
    if st is None or alt is None:
        return df

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_col}:O", title=x_col.capitalize()),
            y=alt.Y(f"{y_col}:O", title=y_col.capitalize()),
            color=alt.Color(f"{value_col}:Q", title=value_col.capitalize(), scale=alt.Scale(scheme="turbo")),
            tooltip=[x_col, y_col, value_col],
        )
    )
    st.subheader(title)
    st.altair_chart(chart, use_container_width=True)
    return chart


def session_heatmap(df: pd.DataFrame, render: bool = True):
    df = df.copy()
    if "session" not in df.columns or "entry_ts" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["session", "date"])["pnl"].sum().reset_index()
    return _heatmap(agg, "date", "session", "pnl", "Session PnL Heatmap", render=render)


def regime_heatmap(df: pd.DataFrame, render: bool = True):
    if "regime" not in df.columns or "entry_ts" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["regime", "date"])["pnl"].sum().reset_index()
    return _heatmap(agg, "date", "regime", "pnl", "Regime PnL Heatmap", render=render)


def tier_heatmap(df: pd.DataFrame, render: bool = True):
    if "tier" not in df.columns or "entry_ts" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["tier", "date"])["pnl"].sum().reset_index()
    return _heatmap(agg, "date", "tier", "pnl", "Tier PnL Heatmap", render=render)
