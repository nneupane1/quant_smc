"""
trade_log.py

TradeZella-style analytics page for:
 - Full trade-by-trade journal
 - Daily and monthly PnL tables
 - Win/Loss statistics
 - Equity curve chart
"""

import streamlit as st
import pandas as pd
import altair as alt

from quant_system.backtest.trade_log import TradeLog
from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter

# Adapter (shared via session_state)
if "dashboard_adapter" not in st.session_state:
    st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()
adapter = st.session_state["dashboard_adapter"]

st.set_page_config(page_title="Trade Log", layout="wide")

# ---------------------------------------------------------
# STYLES
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .positive { color: #00FF9C; }
        .negative { color: #FF4E4E; }
        .neutral  { color: #E0E0E0; }
        .panel {
            padding: 18px;
            border-radius: 12px;
            background: #111417;
            border: 1px solid #222;
        }
        .title {
            font-size: 28px;
            font-weight: 600;
        }
        .subtitle {
            font-size: 18px;
            opacity: 0.8;
        }
        .table-small {
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Trade Log & Analytics</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD TRADES FROM BACKTEST + FORWARD (Unified View)
# ---------------------------------------------------------
def load_all_trades():
    # Backtest trades (persisted)
    try:
        tl = TradeLog()
        df_bt = tl.load_csv()
    except Exception:
        df_bt = pd.DataFrame()

    # Forward trades (live)
    snap = adapter.get_snapshot()
    fwd_events = snap["events"]

    rows = []
    for ev in fwd_events:
        if ev["event"] == "exit_trade":
            p = ev["payload"]
            rows.append({
                "trade_id": ev["trade_id"],
                "timestamp": ev["ts"],
                "side": p.get("side"),
                "entry": p.get("entry_price"),
                "exit": p.get("exit_price"),
                "pnl": p.get("pnl"),
                "r_mult": p.get("r_mult"),
            })

    df_fwd = pd.DataFrame(rows)

    df = pd.concat([df_bt, df_fwd], ignore_index=True)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_all_trades()

if df.empty:
    st.info("No trades available yet.")
    st.stop()

# ---------------------------------------------------------
# METRICS BAR
# ---------------------------------------------------------
wins = df[df["pnl"] > 0]
losses = df[df["pnl"] <= 0]

win_rate = len(wins) / len(df)
avg_r = df["r_mult"].mean()
expectancy = df["pnl"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trades", f"{len(df)}")
col2.metric("Win Rate", f"{win_rate*100:.2f}%")
col3.metric("Avg R-Multiple", f"{avg_r:.2f}")
col4.metric("Expectancy", f"${expectancy:.2f}")


# ---------------------------------------------------------
# EQUITY CURVE
# ---------------------------------------------------------
df_sorted = df.sort_values("timestamp")
df_sorted["equity"] = df_sorted["pnl"].cumsum() + 20_000

chart = (
    alt.Chart(df_sorted)
    .mark_line(color="#00FF9C")
    .encode(
        x="timestamp:T",
        y="equity:Q"
    )
    .properties(height=240)
)

st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------
# TRADE TABLE (highlight winners/losers)
# ---------------------------------------------------------
styled = df.style.apply(
    lambda s: ["color: #00FF9C" if v > 0 
               else "color: #FF4E4E" 
               for v in s] if s.name == "pnl" else [""] * len(s),
    axis=0,
)

st.subheader("Trade List")
st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------
# DAILY STATS
# ---------------------------------------------------------
df["date"] = df["timestamp"].dt.date
daily = df.groupby("date").agg(
    pnl=("pnl", "sum"),
    trades=("pnl", "count"),
    win_rate=("pnl", lambda s: (s > 0).mean()),
    avg_r=("r_mult", "mean"),
).reset_index()

st.subheader("Daily Performance")
st.dataframe(daily, use_container_width=True)


# ---------------------------------------------------------
# MONTHLY STATS
# ---------------------------------------------------------
df["month"] = df["timestamp"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    pnl=("pnl", "sum"),
    trades=("pnl", "count"),
    win_rate=("pnl", lambda s: (s > 0).mean()),
    avg_r=("r_mult", "mean"),
).reset_index()

st.subheader("Monthly Performance")
st.dataframe(monthly, use_container_width=True)


# ---------------------------------------------------------
# R-MULTIPLE DISTRIBUTION
# ---------------------------------------------------------
hist = (
    alt.Chart(df)
    .mark_bar(color="#5ABEFF")
    .encode(x=alt.X("r_mult:Q", bin=True), y="count()")
    .properties(height=200)
)
st.subheader("R-Multiple Distribution")
st.altair_chart(hist, use_container_width=True)
