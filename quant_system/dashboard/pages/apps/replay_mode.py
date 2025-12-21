import streamlit as st
import pandas as pd
import json
import time

from quant_system.backtest.replay_controller import ReplayController
from quant_system.ml.model_registry import ModelRegistry
from quant_system.utils.logger import get_logger
from streamlit.components.v1 import html

LOG = get_logger("replay_mode")

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Replay Mode",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Backtest Replay Engine")


# ------------------------------------------------------------------
# UI Helpers
# ------------------------------------------------------------------

def load_backtest_artifacts(path_prefix: str):
    """Loads candles, SMC, execution logs, model bundle."""
    LOG.info(f"Loading backtest artifacts from {path_prefix}")

    candles = pd.read_csv(f"{path_prefix}/candles_15m.csv", parse_dates=["timestamp"])
    smc = pd.read_csv(f"{path_prefix}/smc_features.csv", parse_dates=["timestamp"])
    exec_log = pd.read_csv(f"{path_prefix}/execution_log.csv")

    registry = ModelRegistry()
    model_bundle = registry.load_all_models(path_prefix)

    conf = registry.load_yaml(f"{path_prefix}/config_used.yaml")

    return candles, smc, exec_log, model_bundle, conf


# ------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Replay Controls")

    bt_path = st.text_input("Backtest folder", "backtest_output/latest")

    if st.button("Load Backtest"):
        st.session_state["loaded"] = True
        st.session_state["bt_artifacts"] = load_backtest_artifacts(bt_path)
        st.success("Backtest loaded.")

    speed = st.slider("Playback Speed (ms per candle)", 150, 1500, 450, 50)

    jump_ts = st.text_input("Jump to timestamp (YYYY-MM-DD HH:MM)")

    col_jump = st.columns(2)
    with col_jump[0]:
        if st.button("Jump"):
            st.session_state["jump_target"] = jump_ts
    with col_jump[1]:
        if st.button("Clear Jump"):
            st.session_state["jump_target"] = None

    st.markdown("---")
    st.header("Actions")
    play_clicked = st.button("Play / Stop")
    next_clicked = st.button("Step →")
    prev_clicked = st.button("← Step")


# ------------------------------------------------------------------
# Load Replay Controller
# ------------------------------------------------------------------

if "loaded" not in st.session_state or not st.session_state["loaded"]:
    st.warning("Load a backtest to begin replay.")
    st.stop()

candles, smc, exec_log, model_bundle, config = st.session_state["bt_artifacts"]

if "controller" not in st.session_state:
    controller = ReplayController(
        candles_15m=candles,
        smc_features=smc,
        execution_log=exec_log,
        model_bundle=model_bundle,
        config=config
    )
    st.session_state["controller"] = controller
else:
    controller = st.session_state["controller"]


# ------------------------------------------------------------------
# TradingView Replay Component (JS)
# ------------------------------------------------------------------
st.subheader("TradingView Replay")

# Load the replay widget
component_html = open("dashboard/streamlit_app/components/tv_replay/index.html").read()
chart_area = html(component_html, height=680, scrolling=False)


# ------------------------------------------------------------------
# REPLAY ACTION LOGIC
# ------------------------------------------------------------------

# Jump to timestamp
if st.session_state.get("jump_target"):
    try:
        ts = pd.to_datetime(st.session_state["jump_target"])
        controller.jump_to_timestamp(ts)
        payload = controller.export_last_json()
        js = f"<script>window.replay_load(`{payload}`);</script>"
        st.markdown(js, unsafe_allow_html=True)
    except Exception as e:
        st.error(str(e))


# Step forward/back
if next_clicked:
    payload = controller.step_forward()
    js = f"<script>window.replay_load(`{json.dumps(payload)}`);</script>"
    st.markdown(js, unsafe_allow_html=True)

if prev_clicked:
    payload = controller.step_backward()
    js = f"<script>window.replay_load(`{json.dumps(payload)}`);</script>"
    st.markdown(js, unsafe_allow_html=True)


# Play / Stop
if play_clicked:
    if "playing" not in st.session_state:
        st.session_state["playing"] = False

    st.session_state["playing"] = not st.session_state["playing"]

# Continuous playback loop
if st.session_state.get("playing", False):
    placeholder = st.empty()
    for _ in range(len(candles)):
        if not st.session_state.get("playing"):
            break

        payload = controller.step_forward()
        js = f"<script>window.replay_load(`{json.dumps(payload)}`);</script>"
        placeholder.markdown(js, unsafe_allow_html=True)

        time.sleep(speed / 1000)


# ------------------------------------------------------------------
# Lower Panel: Meta Information
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("Trade List")

if len(exec_log) > 0:
    st.dataframe(exec_log)
else:
    st.info("No trades recorded in this backtest.")
