"""
Report Builder
Transforms backtest trade logs into daily/monthly summaries and renders Streamlit dashboards.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("report_builder")


# ============================================================
# UTILITY COLORING (green for profit, red for loss)
# ============================================================
def _color_pnl(val):
    if val > 0:
        return "color: #00FF99;"
    if val < 0:
        return "color: #FF4D4D;"
    return "color: white;"


# ============================================================
# KPI PANEL RENDERER
# ============================================================
def _render_kpi_panels(df):
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date
    df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M")

    current_equity = df["pnl"].fillna(0).cumsum().iloc[-1]
    today = df["date"].iloc[-1]
    month = df["month"].iloc[-1]

    today_df = df[df["date"] == today]
    month_df = df[df["month"] == month]

    today_pnl = today_df["pnl"].sum()
    month_pnl = month_df["pnl"].sum()
    win_rate = (df["pnl"] > 0).mean() * 100
    drawdown = df["pnl"].fillna(0).cumsum().cummax() - df["pnl"].fillna(0).cumsum()
    max_dd = drawdown.max()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Current Equity", f"{current_equity:,.2f}")
    c2.metric("Today's PnL", f"{today_pnl:,.2f}")
    c3.metric("This Month PnL", f"{month_pnl:,.2f}")
    c4.metric("Win Rate", f"{win_rate:.2f}%")
    c5.metric("Max Drawdown", f"{max_dd:,.2f}")


# ============================================================
# DAILY TABLE AGGREGATION
# ============================================================
def _daily_report(df):
    df["date"] = pd.to_datetime(df["entry_ts"]).dt.date

    daily = df.groupby("date").agg(
        pnl_sum=("pnl", "sum"),
        trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_r=("r", "mean"),
        best_r=("r", "max"),
        worst_r=("r", "min"),
    ).reset_index()

    return daily


# ============================================================
# MONTHLY TABLE AGGREGATION
# ============================================================
def _monthly_report(df):
    df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M")

    monthly = df.groupby("month").agg(
        pnl_sum=("pnl", "sum"),
        trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_r=("r", "mean"),
        best_r=("r", "max"),
        worst_r=("r", "min"),
    ).reset_index()

    monthly["month"] = monthly["month"].astype(str)
    return monthly


# ============================================================
# LEDGER TABLE (each trade)
# ============================================================
def _trade_ledger(df):
    ledger = df.copy()
    ledger["pnl_color"] = ledger["pnl"]
    return ledger


# ============================================================
# MAIN DASHBOARD
# Called by Backtester when run completes
# ============================================================
def launch_dashboard(trade_log_path: Path):
    """
    Loads the backtest CSV and launches a Streamlit dashboard
    with KPIs, daily/monthly stats, and full trade ledger.
    """
    LOG.info("Launching report dashboard")

    if not trade_log_path.exists():
        LOG.error(f"Trade log missing: {trade_log_path}")
        return

    df = pd.read_csv(trade_log_path, parse_dates=["entry_ts", "exit_ts"])

    if df.empty:
        LOG.error("Trade log empty.")
        return

    st.markdown("""
    <h1>Backtest Results Dashboard</h1>
    <span style='color:#AAA;'>Live Summary • Daily Breakdown • Monthly Breakdown • Ledger</span>
    <hr style='opacity:0.2;margin-top:10px;'>
    """, unsafe_allow_html=True)

    # KPI panels
    _render_kpi_panels(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Daily
    st.subheader("Daily Performance")
    daily = _daily_report(df)
    st.dataframe(
        daily.style.applymap(_color_pnl, subset=["pnl_sum"]),
        height=300,
        use_container_width=True
    )

    # Monthly
    st.subheader("Monthly Performance")
    monthly = _monthly_report(df)
    st.dataframe(
        monthly.style.applymap(_color_pnl, subset=["pnl_sum"]),
        height=300,
        use_container_width=True
    )

    # Full ledger
    st.subheader("Trade Ledger")
    ledger = _trade_ledger(df)
    st.dataframe(
        ledger.style.applymap(_color_pnl, subset=["pnl"]),
        height=500,
        use_container_width=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.success("Backtest report generated successfully.")


# ============================================================
# EXTERNAL API FOR BACKTESTER.PY
# ============================================================
def build_report(df: pd.DataFrame, save_path: Path):
    """
    Used by backtester to save tables before dashboard is launched.
    """

    LOG.info("Building standalone report artifacts")

    daily = _daily_report(df)
    monthly = _monthly_report(df)
    ledger = df.copy()

    # save
    save_path.mkdir(parents=True, exist_ok=True)

    daily.to_csv(save_path / "daily_report.csv", index=False)
    monthly.to_csv(save_path / "monthly_report.csv", index=False)
    ledger.to_csv(save_path / "ledger.csv", index=False)

    LOG.info("Report artifacts saved.")
