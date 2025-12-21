"""
forward_test.py — Hollywood Reasoning Edition

Real-time forward test cockpit with:
 - TV chart (SMC overlays)
 - Equity & risk panels
 - Open trade table
 - Live Confluence reasoning panel
 - EVR, Hazard, Regime, SMC detail blocks
 - Execution log stream
"""

import time
import streamlit as st
import pandas as pd
import altair as alt

from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter
from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart

if "dashboard_adapter" not in st.session_state:
    st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()
adapter = st.session_state["dashboard_adapter"]

st.set_page_config(page_title="Forward Test Cockpit", layout="wide")

# ---------------------------------------------------------
# STYLES
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .panel {
            padding: 20px;
            border-radius: 12px;
            background: #111417;
            border: 1px solid #222;
        }
        .metric-large { font-size: 38px; font-weight: 600; }
        .metric-small { font-size: 14px; opacity: 0.8; }
        .positive { color: #00FF9C; }
        .negative { color: #FF4E4E; }
        .neutral  { color: #CACACA; }
        .reason-title {
            font-size: 20px;
            font-weight: 600;
            padding-bottom: 6px;
        }
        .reason-block {
            background: #1A1D21;
            padding: 14px;
            border-left: 4px solid #5ABEFF;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .reason-label {
            font-weight: 600;
            margin-bottom: 4px;
            color: #A8C8FF;
        }
        .codebox {
            font-size: 13px;
            background: #0E1012;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #222;
            color: #D6D6D6;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE LAYOUT
# =========================================================

col_left, col_right = st.columns([2, 1])

snap = adapter.get_snapshot()
state = snap["state"]
events = snap["events"]

# =========================================================
# LEFT COLUMN
# =========================================================

with col_left:

    st.title("Forward Test – Live Cockpit")

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------
    st.subheader("BTC/USD Live Chart")
    render_tv_chart(snap["candles"], key="chart")

    # ---------------------------------------------------------
    # EXECUTION LOG
    # ---------------------------------------------------------
    st.subheader("Execution Log")
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("No execution events yet.")


# =========================================================
# RIGHT COLUMN — Reasoning Engine Display
# =========================================================

with col_right:

    st.subheader("Account Overview")
    equity = state.get("equity", 0)
    locked = state.get("locked_profit", 0)
    free = state.get("free_capital", 0)

    pnl_color = "positive" if equity >= 20_000 else "negative"

    st.markdown(
        f"""
        <div class="panel">
            <div class="metric-large {pnl_color}">${equity:,.2f}</div>
            <div class="metric-small">Current Equity</div>
            <br/>

            <div class="metric-large positive">${locked:,.2f}</div>
            <div class="metric-small">Locked Profit</div>
            <br/>

            <div class="metric-large neutral">${free:,.2f}</div>
            <div class="metric-small">Free Capital</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Open Trades")
    if state.get("open_trades"):
        st.dataframe(pd.DataFrame(state["open_trades"]), use_container_width=True)
    else:
        st.info("No open trades.")

    # ---------------------------------------------------------
    # Confluence Reasoning Panel — NEW
    # ---------------------------------------------------------
    st.subheader("Confluence Reasoning Engine")

    if not events:
        st.info("No decisions yet.")
    else:
        # We take the last event with reasoning payload
        ev = events[-1]
        payload = ev["payload"]

        reason = payload.get("reasoning", {})

        # 1 — SMC reasoning
        smc = reason.get("smc", {})
        with st.expander("SMC Context", expanded=True):
            st.markdown('<div class="reason-block">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="reason-label">Swings</div>
                <div class="codebox">{smc.get("swings")}</div>
                <div class="reason-label">BOS/CHOCH</div>
                <div class="codebox">{smc.get("bos")}</div>
                <div class="reason-label">Liquidity Sweeps</div>
                <div class="codebox">{smc.get("sweeps")}</div>
                <div class="reason-label">Zones</div>
                <div class="codebox">{smc.get("zones")}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 2 — ML specialists
        ml = reason.get("ml", {})
        with st.expander("Model Probabilities", expanded=True):
            for k, v in ml.items():
                st.markdown(
                    f"""
                    <div class="reason-block">
                        <div class="reason-label">{k}</div>
                        <div class="codebox">{v}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3 — Regime
        regime = reason.get("regime", {})
        with st.expander("Regime (HMM/HDBSCAN)", expanded=False):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">State Probabilities</div>
                <div class="codebox">{regime}</div>
            </div>
            """, unsafe_allow_html=True)

        # 4 — EMA alignment
        ema = reason.get("ema", {})
        with st.expander("EMA Alignment", expanded=False):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">EMA Signals</div>
                <div class="codebox">{ema}</div>
            </div>
            """, unsafe_allow_html=True)

        # 5 — Confluence breakdown
        conf = reason.get("confluence_breakdown", {})
        with st.expander("Confluence Breakdown", expanded=True):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">Weights × Probabilities</div>
                <div class="codebox">{conf}</div>
            </div>
            """, unsafe_allow_html=True)

        # 6 — EVR
        evr = reason.get("evr", {})
        with st.expander("EVR Details", expanded=False):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">Expected R Calculations</div>
                <div class="codebox">{evr}</div>
            </div>
            """, unsafe_allow_html=True)

        # 7 — Hazard
        hazard = reason.get("hazard", {})
        with st.expander("Hazard Snapshot (Survival Model)", expanded=False):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">Hazard Value</div>
                <div class="codebox">{hazard}</div>
            </div>
            """, unsafe_allow_html=True)

        # 8 — Final decision
        final = reason.get("final_decision", {})
        with st.expander("Final Execution Decision", expanded=True):
            st.markdown(f"""
            <div class="reason-block">
                <div class="reason-label">Decision</div>
                <div class="codebox">{final}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------
time.sleep(0.3)
