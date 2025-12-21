import streamlit as st
from streamlit.components.v1 import html
import json
import time
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("live_monitor")


# JS bundle loader (TradingView + overlays + panels)
def load_chart_component(height=620):
    tv_path = Path(__file__).parents[2] / "components" / "js" / "tv_chart_build" / "index.html"
    with open(tv_path, "r") as f:
        html_file = f.read()
    html(html_file, height=height, scrolling=False)


# WebSocket-like bridge using Streamlit session_state
def push_payload_to_js(payload: dict):
    """
    Injects payload into the JS side of the chart.
    """
    js = f"""
        <script>
            window.tv_chart_update({json.dumps(payload)});
        </script>
    """
    st.markdown(js, unsafe_allow_html=True)


def _header_section():
    st.markdown(
        """
        <h1 style="margin-bottom:0px;">Live Trading Cockpit</h1>
        <span style="color:#888;">Execution • ML Reasoning • Regime • Risk Engine</span>
        <hr style="margin-top:12px;margin-bottom:20px;opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )


def _top_status_bar(theme, model_version):
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div style="font-size:18px;">
                Theme: <b>{theme}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="font-size:18px;">
                Model Version: <b>{model_version}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        trading_enabled = st.toggle("Live Trading", value=False)
        st.session_state["live_trading_enabled"] = trading_enabled

    with col4:
        lev_on = st.toggle("Leverage", value=False)
        st.session_state["live_leverage"] = lev_on


def _side_panels():
    st.markdown(
        """
        <style>
        .side-panel-box {
            padding:12px;
            background:#111;
            border:1px solid #444;
            border-radius:6px;
            margin-bottom:12px;
        }
        .side-panel-title {
            font-weight:600;
            font-size:17px;
            margin-bottom:4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="side-panel-title">Confluence</div>', unsafe_allow_html=True)
    st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="side-panel-title">Hazard Timeline</div>', unsafe_allow_html=True)
    st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="side-panel-title">Regime</div>', unsafe_allow_html=True)
    st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="side-panel-title">Risk & MPC</div>', unsafe_allow_html=True)
    st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="side-panel-title">Current Position</div>', unsafe_allow_html=True)
    st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


def render_live(theme_choice: str, model_version: str):
    _header_section()
    _top_status_bar(theme_choice, model_version)

    chart_col, side_col = st.columns([4, 1])

    with chart_col:
        load_chart_component()

    with side_col:
        _side_panels()

    # Real-time stream simulation (to be replaced with orchestrator)
    placeholder = st.empty()

    # Example: Sustained live payload from orchestrator
    if "live_payload" not in st.session_state:
        st.session_state.live_payload = None

    # Fake loop (orchestrator will update session_state['live_payload'])
    for _ in range(1):  # run once, JS stays alive
        payload = st.session_state.live_payload
        if payload:
            LOG.info("Pushing live payload to JS")
            push_payload_to_js(payload)
        time.sleep(0.1)

    st.markdown(
        """
        <script>
        // keep JS alive, allow external orchestrator to push events
        console.log("Live monitor initialized.");
        </script>
        """,
        unsafe_allow_html=True,
    )
