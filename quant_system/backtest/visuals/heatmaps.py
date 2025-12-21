import streamlit as st
import pandas as pd
import altair as alt


def _heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str):
    if df.empty:
        st.write("No data available.")
        return

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


def session_heatmap(df: pd.DataFrame):
    df = df.copy()
    if "session" not in df.columns:
        return
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["session", "date"])["pnl"].sum().reset_index()
    _heatmap(agg, "date", "session", "pnl", "Session PnL Heatmap")


def regime_heatmap(df: pd.DataFrame):
    if "regime" not in df.columns:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["regime", "date"])["pnl"].sum().reset_index()
    _heatmap(agg, "date", "regime", "pnl", "Regime PnL Heatmap")


def tier_heatmap(df: pd.DataFrame):
    if "tier" not in df.columns:
        return
    df = df.copy()
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    agg = df.groupby(["tier", "date"])["pnl"].sum().reset_index()
    _heatmap(agg, "date", "tier", "pnl", "Tier PnL Heatmap")
