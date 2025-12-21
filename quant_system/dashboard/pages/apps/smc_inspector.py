import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from streamlit.components.v1 import html
from quant_system.utils.logger import get_logger

LOG = get_logger("smc_inspector")

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="SMC Inspector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Smart Money Concepts Inspector")


# --------------------------------------------------------------
# Load replay context
# --------------------------------------------------------------
if "controller" not in st.session_state:
    st.warning("Replay controller is not loaded. Load a backtest first.")
    st.stop()

controller = st.session_state["controller"]
df = controller.df
smc = controller.smc


# --------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    candle_idx = st.number_input(
        "Candle Index", 
        min_value=0,
        max_value=len(df)-1,
        value=int(len(df)//2),
        step=1
    )

    if st.button("Jump in Replay"):
        ts = int(df.loc[candle_idx, "timestamp"].timestamp())
        js = f"<script>ReplayWidget.send('jump', {{timestamp:{ts}}});</script>"
        st.markdown(js, unsafe_allow_html=True)

    st.markdown("---")

    show_ob = st.checkbox("Show Orderblocks", True)
    show_fvg = st.checkbox("Show FVGs", True)
    show_bos = st.checkbox("Show BOS/CHOCH", True)
    show_sweeps = st.checkbox("Show Sweeps", True)
    show_swings = st.checkbox("Show Swings", True)


# --------------------------------------------------------------
# Extract SMC data for chosen candle
# --------------------------------------------------------------
row = df.iloc[candle_idx]
smc_row = smc.iloc[candle_idx]

ts = row["timestamp"]
st.subheader(f"Structure at {ts}")

# Decode JSON fields
def parse(field):
    try:
        return json.loads(smc_row.get(field, "[]"))
    except Exception:
        return []

swings = parse("swings_json")
orderblocks = parse("ob_json")
fvg = parse("fvg_json")
sweeps = parse("sweeps_json")
bos_choch = parse("bos_choch_json")


# --------------------------------------------------------------
# Multi-pane TradingView SMC display
# --------------------------------------------------------------

def load_component():
    code = open(Path(__file__).parents[2] / "components" / "js" / "tv_chart_build" / "index.html").read()
    return html(code, height=720, scrolling=False)

st.subheader("Chart View with SMC Overlays")
load_component()


# Stream overlays into JS
payload = {
    "candle": {
        "time": int(ts.timestamp()),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0)),
    },
    "smc": {
        "swings": swings if show_swings else [],
        "orderblocks": orderblocks if show_ob else [],
        "fvg": fvg if show_fvg else [],
        "sweeps": sweeps if show_sweeps else [],
        "bos_choch": bos_choch if show_bos else []
    },
    "trade": {"entries": [], "exits": [], "stops": [], "hedge": []},
    "meta": {"conf": np.nan, "evr": np.nan, "hazard": np.nan, "risk": {}}
}

js_call = f"<script>window.replay_load(`{json.dumps(payload)}`);</script>"
st.markdown(js_call, unsafe_allow_html=True)


# --------------------------------------------------------------
# STRUCTURE TABLE
# --------------------------------------------------------------
st.markdown("---")
st.subheader("Structure Extract")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Swings")
    st.json(swings if show_swings else [])

with col2:
    st.markdown("### Orderblocks")
    st.json(orderblocks if show_ob else [])

with col3:
    st.markdown("### FVG Zones")
    st.json(fvg if show_fvg else [])


# --------------------------------------------------------------
# BOS/CHOCH timeline
# --------------------------------------------------------------
st.markdown("---")
st.subheader("BOS / CHOCH Timeline Window")

# Look ±100 candles around chosen index
w = 100
start = max(0, candle_idx - w)
end = min(len(df)-1, candle_idx + w)

events = []
for i in range(start, end+1):
    evlist = json.loads(smc.iloc[i].get("bos_choch_json", "[]"))
    for ev in evlist:
        events.append({
            "idx": i,
            "time": str(df.loc[i, "timestamp"]),
            "type": ev["type"],
            "price": ev["price"]
        })

ev_df = pd.DataFrame(events)
if len(ev_df) > 0:
    st.dataframe(ev_df)
else:
    st.info("No BOS/CHOCH events in the window.")


# --------------------------------------------------------------
# OB / FVG Density Map Visualization
# --------------------------------------------------------------
st.markdown("---")
st.subheader("Orderblock / FVG Density Map (Recent 500 bars)")

def density_map(json_col, N=500):
    arr = []
    for i in range(max(0, len(smc)-N), len(smc)):
        lst = json.loads(smc.iloc[i].get(json_col, "[]"))
        arr.append(len(lst))
    return np.array(arr)

ob_density = density_map("ob_json")
fvg_density = density_map("fvg_json")

colA, colB = st.columns(2)
with colA:
    st.line_chart(ob_density)
    st.caption("Orderblock density (last 500 bars)")

with colB:
    st.line_chart(fvg_density)
    st.caption("FVG density (last 500 bars)")


# --------------------------------------------------------------
# Feature Vector Inspection
# --------------------------------------------------------------
st.markdown("---")
st.subheader("Feature Vector Inspection")

feat_cols = [c for c in df.columns if c.startswith("feat_")]
feat_vec = df.loc[candle_idx, feat_cols]

feat_dict = {col: float(feat_vec[col]) for col in feat_cols}
st.json(feat_dict)


# --------------------------------------------------------------
# End
# --------------------------------------------------------------
