from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from quant_system.backtest.replay_controller import ReplayController
from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import page_header, section_title
from quant_system.ml.registry.model_registry import ModelRegistry


def _registry(context: DashboardContext):
    try:
        if context.model_dir.exists():
            return ModelRegistry(str(context.model_dir))
    except Exception:
        return None
    return None


def _controller(context: DashboardContext) -> ReplayController:
    key = ("replay_controller", str(context.backtest_dir), str(context.model_dir))
    cached_key = st.session_state.get("replay_controller_key")
    if cached_key == key and "replay_controller" in st.session_state:
        return st.session_state["replay_controller"]

    controller = ReplayController(
        candles_15m=context.backtest["candles"],
        smc_features=context.backtest["smc_features"],
        execution_log=context.backtest["execution_log"] if not context.backtest["execution_log"].empty else context.backtest["trades"],
        model_bundle=_registry(context),
        config=context.config,
    )
    st.session_state["replay_controller"] = controller
    st.session_state["replay_controller_key"] = key
    st.session_state["replay_index"] = 0
    return controller


def render_replay_mode(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    if context.backtest["candles"].empty:
        st.info("Replay requires saved candles in the active backtest directory.")
        return

    page_header(
        "Replay Mode",
        "Candle-by-candle replay on top of the repaired backtest artifacts and current predictor stack.",
        kicker="Replay Engine",
    )
    controller = _controller(context)
    candles = context.backtest["candles"].copy()
    candles["timestamp"] = pd.to_datetime(
        candles["timestamp"] if "timestamp" in candles.columns else candles["dt"], errors="coerce"
    )
    max_idx = max(len(candles) - 1, 0)

    controls = st.columns([1.1, 1.1, 1.1, 2.1])
    if controls[0].button("Step Back", use_container_width=True):
        controller.step_backward()
    if controls[1].button("Step Forward", use_container_width=True):
        controller.step_forward()
    playing = controls[2].toggle("Auto Play", value=False)
    speed_ms = controls[3].slider("Playback ms/bar", min_value=100, max_value=1500, value=350, step=50)

    idx = st.slider("Replay Index", min_value=0, max_value=max_idx, value=int(controller.state.ptr), step=1)
    if idx != controller.state.ptr:
        controller.jump_to(idx)

    if playing:
        time.sleep(speed_ms / 1000.0)
        controller.step_forward()
        st.rerun()

    payload = controller.render_payload()
    window_start = max(0, controller.state.ptr - 120)
    render_tv_chart(candles.iloc[window_start: controller.state.ptr + 1].copy(), key="replay_window")

    left, right = st.columns([1.3, 1.1])
    with left:
        section_title("Replay Payload", "What the chart/reasoning layer sees at this step")
        st.json(payload)
    with right:
        section_title("Events At Cursor", "Entries, exits, stops, and hedge events")
        st.json(payload.get("trade", {}))
