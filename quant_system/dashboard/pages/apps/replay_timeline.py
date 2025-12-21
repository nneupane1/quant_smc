import streamlit as st
import pandas as pd
import numpy as np
import json
from quant_system.backtest.replay_controller import ReplayController
from quant_system.utils.logger import get_logger
from streamlit.components.v1 import html

LOG = get_logger("replay_timeline")

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Replay Timeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Replay Timeline Navigator")


# --------------------------------------------------------------
# Load cached replay controller
# --------------------------------------------------------------
if "controller" not in st.session_state:
    st.warning("Backtest not loaded. Go to Replay Mode first.")
    st.stop()

controller: ReplayController = st.session_state["controller"]
df = controller.df
exec_log = controller.exec


# --------------------------------------------------------------
# Build timeline data
# --------------------------------------------------------------
df["t"] = df["timestamp"].astype("int64") // 10**9

# Compute summary metrics per candle (these were computed in replay_controller)
def safe_extract(meta_col):
    return df[meta_col] if meta_col in df.columns else np.zeros(len(df))

# For timeline heatmap visualization
if "conf" not in df.columns:
    df["conf"] = np.nan
if "evr" not in df.columns:
    df["evr"] = np.nan
if "hazard" not in df.columns:
    df["hazard"] = np.nan


# --------------------------------------------------------------
# Sidebar: Zoom + Filters
# --------------------------------------------------------------
with st.sidebar:
    st.header("Timeline Options")

    zoom = st.select_slider(
        "Zoom Level",
        options=["15m", "1h", "4h", "1d", "1w"],
        value="4h"
    )

    metric = st.selectbox("Heatmap Metric", ["conf", "evr", "hazard"])

    st.markdown("---")
    st.write("Trade Markers:")
    show_entries = st.checkbox("Entries", True)
    show_exits = st.checkbox("Exits", True)
    show_stops = st.checkbox("Stops", True)


# --------------------------------------------------------------
# Resampling for zoom levels
# --------------------------------------------------------------
rule_map = {
    "15m": "15T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W"
}

rule = rule_map[zoom]

zdf = df.resample(rule, on="timestamp").agg({
    "conf": "mean",
    "evr": "mean",
    "hazard": "mean",
    "open": "first",
    "close": "last",
})

zdf = zdf.dropna(subset=["open", "close"]).reset_index()
zdf["t"] = zdf["timestamp"].astype("int64") // 10**9

metric_vals = zdf[metric].fillna(0).values
norm = (metric_vals - metric_vals.min()) / (metric_vals.ptp() + 1e-6)

# Color mapping for heatmap
colors = [
    f"rgba({int(255*n)}, {int(50)}, {int(255*(1-n))}, 0.75)"
    for n in norm
]


# --------------------------------------------------------------
# Build timeline HTML panel (D3-free, simple, flickerless)
# --------------------------------------------------------------

def make_timeline_html():
    bars = []
    for i, row in zdf.iterrows():
        timestamp = int(row["t"])
        color = colors[i]
        title = f"{row['timestamp']} | {metric.upper()}={row[metric]:.3f}"

        bars.append(f"""
            <div class="bar" 
                 style="background:{color}" 
                 title="{title}" 
                 onclick="ReplayWidget.send('jump', {{timestamp:{timestamp}}})">
            </div>
        """)

    html_str = f"""
    <style>
        .timeline {{
            width: 100%;
            height: 60px;
            display:flex;
            flex-direction:row;
            border:1px solid #333;
            background:#111;
            overflow:hidden;
        }}
        .bar {{
            flex:1;
            cursor:pointer;
            transition:opacity 0.2s ease;
        }}
        .bar:hover {{
            opacity:0.7;
        }}
    </style>

    <div class="timeline">
        {''.join(bars)}
    </div>
    """
    return html_str


# Render timeline
st.subheader("Timeline Heatmap")
st.markdown(make_timeline_html(), unsafe_allow_html=True)


# --------------------------------------------------------------
# TRADE TABLE (click row to jump to replay)
# --------------------------------------------------------------
st.markdown("---")
st.subheader("Trades")

if len(exec_log) == 0:
    st.info("No trades in this backtest.")
else:
    # Clickable rows
    exec_log = exec_log.copy()
    exec_log["jump"] = exec_log["candle_idx"].apply(
        lambda i: f"<a href='#' onclick=\"ReplayWidget.send('jump', {{timestamp:{int(df.loc[i,'t'])}}})\">Go</a>"
    )
    st.write(
        exec_log.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )


# --------------------------------------------------------------
# END OF FILE
# --------------------------------------------------------------
