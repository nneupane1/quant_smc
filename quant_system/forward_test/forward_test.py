"""
forward_test.py

Real-time forward test cockpit.
Displays:
 - Live BTC price chart with SMC overlays (TV chart)
 - Account equity, locked profit, free capital
 - Open trades table
 - Confluence / EVR / Hazard gauges
 - Live execution logs (entries, exits, trail updates)
"""

import time
import streamlit as st
import pandas as pd

from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter
from dashboard.streamlit_app.components.tv_chart import render_tv_chart

# Shared adapter object (should be created by Streamlit entrypoint app.py)
# Here we load from st.session_state
if "dashboard_adapter" not in st.session_state:
    st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()
adapter = st.session_state["dashboard_adapter"]

st.set_page_config(
    page_title="Forward Test Cockpit",
    layout="wide",
)

# ----------------------------------------------------------------------
# STYLES (light aesthetic)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .metric-large {font-size: 38px; font-weight: 600; padding-bottom: 4px;}
        .metric-small {font-size: 14px; opacity: 0.75;}
        .positive {color: #00FF9C;}
        .negative {color: #FF4E4E;}
        .neutral  {color: #E0E0E0;}
        .panel {
            padding: 18px; 
            border-radius: 10px; 
            background: #111417; 
            border: 1px solid #222;
        }
        .table-small {
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------------------------
st.title("Forward Test – Live Trading Simulator")


# ----------------------------------------------------------------------
# AUTO-REFRESH (non-blocking)
# ----------------------------------------------------------------------
refresh_rate_ms = 300  # ~3 FPS, smooth enough

# ----------------------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------------------
col_equity, col_chart = st.columns([1, 2])

# ----------------------------------------------------------------------
# EQUITY PANEL
# ----------------------------------------------------------------------
with col_equity:
    st.subheader("Account Overview")

    snap = adapter.get_snapshot()
    state = snap["state"]

    equity = state.get("equity", 0)
    locked_profit = state.get("locked_profit", 0)
    free_capital = state.get("free_capital", 0)
    hedge_ratio = state.get("hedge_ratio", 0)
    risk_mode = state.get("risk_mode", "N/A")
    cooling_end = state.get("cooling_end")

    pnl_color = "positive" if equity >= 20_000 else "negative"

    st.markdown(
        f"""
        <div class="panel">
            <div class="metric-large {pnl_color}">${equity:,.2f}</div>
            <div class="metric-small">Current Equity</div>

            <div class="metric-large positive">${locked_profit:,.2f}</div>
            <div class="metric-small">Locked Profit (Vault)</div>

            <div class="metric-large neutral">${free_capital:,.2f}</div>
            <div class="metric-small">Free Capital (Available)</div>

            <div class="metric-large neutral">{risk_mode}</div>
            <div class="metric-small">Risk Mode</div>

            <div class="metric-large neutral">{hedge_ratio:.2f}</div>
            <div class="metric-small">Hedge Ratio</div>

            <div class="metric-large neutral">
                {cooling_end if cooling_end else "—"}
            </div>
            <div class="metric-small">Cooling Ends</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Open Trades")

    trades = state.get("open_trades", [])
    if trades:
        df_open = pd.DataFrame(trades)
        st.dataframe(df_open, use_container_width=True)
    else:
        st.info("No open trades at the moment.")


# ----------------------------------------------------------------------
# CHART PANEL – TradingView JS Chart
# ----------------------------------------------------------------------
with col_chart:
    st.subheader("Live BTC/USD Chart")

    candles = snap["candles"]
    render_tv_chart(candles, key="forward_tv_chart")


# ----------------------------------------------------------------------
# METRICS BAR – Confluence, EVR, Hazard (from last event)
# ----------------------------------------------------------------------
st.subheader("Live Signals")

events = snap["events"]
if events:
    # Look at last event only, if applicable
    last_event = events[-1]
    payload = last_event["payload"]

    # If row contains execution metrics
    conf_score = payload.get("conf_score")
    evr = payload.get("evr")
    hazard = payload.get("hazard")
    side = payload.get("side")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("### Confluence")
        st.progress(min(max(conf_score or 0, 0), 1.0))

    with col_b:
        st.markdown("### Expected-R")
        st.progress(min(max((evr or 0) / 3.0, 0), 1.0))

    with col_c:
        st.markdown("### Hazard")
        st.progress(min(max(hazard or 0, 0), 1.0))

    with col_d:
        st.markdown("### Side")
        st.success(side if side else "N/A")

else:
    st.info("No recent execution events.")


# ----------------------------------------------------------------------
# EXECUTION LOGS TABLE
# ----------------------------------------------------------------------
st.subheader("Execution Log")

if events:
    df_events = pd.DataFrame(events)
    st.dataframe(df_events, use_container_width=True)
else:
    st.info("No events logged yet.")


# ----------------------------------------------------------------------
# AUTO REFRESH
# ----------------------------------------------------------------------
st_autorefresh = st.experimental_memo.clear()


time.sleep(refresh_rate_ms / 1000.0)
