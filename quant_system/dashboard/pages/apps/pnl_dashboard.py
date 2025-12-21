import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("pnl_dashboard")


def _load_backtest_report():
    """
    Loads the most recent backtest JSON report + trade log CSV.
    """
    base = Path.cwd() / "backtest_outputs"

    report_file = base / "summary.json"
    trades_file = base / "trades.csv"

    if not report_file.exists() or not trades_file.exists():
        return None, None

    with open(report_file, "r") as f:
        report = json.load(f)

    trades = pd.read_csv(trades_file)
    return report, trades


def _section_title(label):
    st.markdown(
        f"""
        <h2 style="margin-bottom:4px;">{label}</h2>
        <hr style="margin-top:6px;margin-bottom:16px;opacity:0.2;">
        """,
        unsafe_allow_html=True,
    )


def _equity_curve(report):
    df = pd.DataFrame(report["equity_curve"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")

    st.line_chart(df["equity"], height=280)


def _pnl_breakdowns(trades):
    trades["date"] = pd.to_datetime(trades["entry_time"]).dt.date
    daily = trades.groupby("date")["pnl"].sum()

    st.subheader("Daily PnL")
    st.bar_chart(daily, height=200)

    month = trades.groupby(trades["date"].astype("datetime64[M]"))["pnl"].sum()
    st.subheader("Monthly PnL")
    st.bar_chart(month, height=180)


def _drawdown_plot(report):
    dd = pd.DataFrame(report["drawdown"])
    dd["time"] = pd.to_datetime(dd["time"], unit="s")
    dd = dd.set_index("time")

    st.subheader("Drawdown Timeline")
    st.area_chart(dd["dd"], height=180)


def _regime_perf(report):
    regime_series = report["regime_performance"]
    df = pd.DataFrame(regime_series)

    st.subheader("Regime Performance")
    st.dataframe(df, height=200)


def _evr_vs_r_scatter(trades):
    st.subheader("EVR vs Realized R")
    import altair as alt

    chart = (
        alt.Chart(trades)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x="evr:Q",
            y="realized_r:Q",
            color=alt.Color("result:N", scale=alt.Scale(domain=["win", "loss"],
                                                       range=["#1fdf8f", "#ff4b5c"])),
            tooltip=["entry_time", "side", "evr", "realized_r"]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _trade_table(trades):
    st.subheader("Trades")
    st.dataframe(
        trades[["entry_time", "exit_time", "side", "evr", "conf", "realized_r", "pnl"]],
        height=340
    )


# -------------------------------------------------------------------
# HOME PAGE RENDER
# -------------------------------------------------------------------
def render_home(theme_choice, model_version):
    LOG.info("Rendering Home Dashboard")

    st.markdown(
        f"""
        <h1>Home Dashboard</h1>
        <span style="color:#888;">
            Overview • Equity • Performance • Live Stats
        </span>
        <hr style="margin-top:12px;margin-bottom:20px;opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )

    report, trades = _load_backtest_report()

    if report is None:
        st.warning("No backtest results found.")
        return

    _section_title("Equity Curve")
    _equity_curve(report)

    col1, col2 = st.columns([1, 1])
    with col1:
        _section_title("Drawdown")
        _drawdown_plot(report)

    with col2:
        _section_title("Regime Performance")
        _regime_perf(report)

    if trades is not None:
        _section_title("Daily / Monthly PnL")
        _pnl_breakdowns(trades)


# -------------------------------------------------------------------
# BACKTEST RESULTS PAGE RENDER
# -------------------------------------------------------------------
def render_backtest(theme_choice, model_version):
    LOG.info("Rendering Backtest Dashboard")

    st.markdown(
        f"""
        <h1>Backtest Results</h1>
        <span style="color:#888;">Analysis • Trades • Equity • Diagnostics</span>
        <hr style="margin-top:12px;margin-bottom:20px;opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )

    report, trades = _load_backtest_report()

    if report is None or trades is None:
        st.warning("No backtest output found.")
        return

    _section_title("Equity Curve")
    _equity_curve(report)

    col1, col2 = st.columns([1, 1])
    with col1:
        _section_title("Drawdown")
        _drawdown_plot(report)
    with col2:
        _section_title("Regime Performance")
        _regime_perf(report)

    _section_title("EVR vs Realized R")
    _evr_vs_r_scatter(trades)

    _section_title("Trade List")
    _trade_table(trades)
