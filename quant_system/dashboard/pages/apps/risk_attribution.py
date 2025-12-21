import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from streamlit.components.v1 import html

from quant_system.utils.logger import get_logger

LOG = get_logger("risk_attribution")


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Attribution",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Risk Attribution Dashboard")


# ---------------------------------------------------------------------
# Load backtest context
# ---------------------------------------------------------------------
if "controller" not in st.session_state or "bt_artifacts" not in st.session_state:
    st.warning("Please load a backtest in Replay Mode first.")
    st.stop()

controller = st.session_state["controller"]
candles, smc, exec_log, model_bundle, config = st.session_state["bt_artifacts"]

df = controller.df.copy()
exec_df = controller.exec.copy()

if "equity" not in df.columns:
    st.error("Backtester must output equity curve in df['equity']")
    st.stop()


# ---------------------------------------------------------------------
# SECTION 1 — EQUITY CURVE + DRAWDOWN
# ---------------------------------------------------------------------
st.subheader("1. Equity Curve & Drawdown")

df["dd"] = df["equity"] - df["equity"].cummax()
df["dd_pct"] = df["dd"] / df["equity"].cummax()

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["equity"],
        mode="lines",
        line=dict(color="#4FC3F7", width=2),
        name="Equity"
    ))
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["dd_pct"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#EF5350"),
        name="Drawdown %"
    ))
    dd_fig.update_layout(height=350, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(dd_fig, use_container_width=True)


# ---------------------------------------------------------------------
# SECTION 2 — FACTOR ATTRIBUTION
# ---------------------------------------------------------------------
"""
We attribute realized returns to:
 - Trend regime (HMM)
 - Range regime
 - Expansion regime
 - Collapse regime
 - Hazard exits
 - MPC hedge contribution
 - Exposure (net/gross)
"""

st.markdown("---")
st.subheader("2. Factor Attribution")

# Needed fields:
req_cols = ["return_r", "regime_trend_p", "regime_range_p", "regime_expansion_p",
            "regime_collapse_p", "hazard", "hedge_ratio"]

missing = [c for c in req_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns for risk attribution: {missing}")
    st.stop()

# Compute factor exposures
df["trend_contrib"] = df["return_r"] * df["regime_trend_p"]
df["range_contrib"] = df["return_r"] * df["regime_range_p"]
df["exp_contrib"] = df["return_r"] * df["regime_expansion_p"]
df["col_contrib"] = df["return_r"] * df["regime_collapse_p"]

# Hazard-induced losses:
df["hazard_contrib"] = -np.abs(df["return_r"]) * df["hazard"].clip(0, 1)

# Hedge contribution:
df["hedge_contrib"] = df["return_r"] * (df["hedge_ratio"].fillna(0))

factor_totals = {
    "Trend": df["trend_contrib"].sum(),
    "Range": df["range_contrib"].sum(),
    "Expansion": df["exp_contrib"].sum(),
    "Collapse": df["col_contrib"].sum(),
    "Hazard": df["hazard_contrib"].sum(),
    "Hedge": df["hedge_contrib"].sum(),
}

# PIE CHART
fig_f = go.Figure(data=[go.Pie(
    labels=list(factor_totals.keys()),
    values=list(factor_totals.values()),
    hole=0.4
)])
fig_f.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_f, use_container_width=True)


# ---------------------------------------------------------------------
# SECTION 3 — CVaR Attribution
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("3. CVaR₀.₉₅ Attribution")

# Compute tail distribution
returns = df["return_r"].dropna()
cutoff = np.quantile(returns, 0.05)

tail_losses = returns[returns <= cutoff]

# CVaR by factor exposure
cvar_map = {
    "Trend": (df.loc[returns <= cutoff, "trend_contrib"]).sum(),
    "Range": (df.loc[returns <= cutoff, "range_contrib"]).sum(),
    "Expansion": (df.loc[returns <= cutoff, "exp_contrib"]).sum(),
    "Collapse": (df.loc[returns <= cutoff, "col_contrib"]).sum(),
    "Hazard": (df.loc[returns <= cutoff, "hazard_contrib"]).sum(),
    "Hedge": (df.loc[returns <= cutoff, "hedge_contrib"]).sum(),
}

fig_c = go.Figure()
fig_c.add_trace(go.Bar(
    x=list(cvar_map.keys()),
    y=list(cvar_map.values()),
    marker_color=["#4FC3F7", "#81C784", "#FFD54F", "#E57373", "#F06292", "#64B5F6"]
))
fig_c.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_c, use_container_width=True)


# ---------------------------------------------------------------------
# SECTION 4 — Hazard Exit Attribution Timeline
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Hazard Exit Attribution Timeline")

haz_fig = go.Figure()
haz_fig.add_trace(go.Bar(
    x=df["timestamp"],
    y=df["hazard_contrib"],
    marker_color="#F06292"
))
haz_fig.update_layout(
    template="plotly_dark",
    height=350,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(haz_fig, use_container_width=True)


# ---------------------------------------------------------------------
# SECTION 5 — Exposure Attribution (Net/Gross/Hedge)
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("5. Exposure Attribution")

if "net_exposure" not in df.columns:
    df["net_exposure"] = df["hedge_ratio"] * -1  # fallback

exp_fig = go.Figure()
exp_fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["net_exposure"],
    mode="lines",
    line=dict(color="#4FC3F7", width=2),
    name="Net Exposure"
))
exp_fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["hedge_ratio"],
    mode="lines",
    line=dict(color="#FFB300", width=2),
    name="Hedge Ratio"
))

exp_fig.update_layout(
    template="plotly_dark",
    height=350,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(exp_fig, use_container_width=True)


# ---------------------------------------------------------------------
# SECTION 6 — Jump to Replay from Risk Events
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("6. Drill Down Into Event in Replay")

# Which candle?
idx_select = st.number_input(
    "Candle index for replay jump",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

if st.button("Jump to Replay"):
    ts = int(df.loc[idx_select, "timestamp"].timestamp())
    js = f"<script>ReplayWidget.send('jump', {{timestamp:{ts}}})</script>"
    st.markdown(js, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------
