"""
replay_control_bar.py
Full replay control interface with:
 • Play/pause
 • Step forward/backward
 • Speed slider
 • Timeline scrubber
 • Jump-to-event navigation
 • Asset focus switching
 • JS postMessage broadcast

Integrates with:
 • ReplayEngine
 • ReplayTimeline
 • Dashboard (Streamlit + JS)
"""

import streamlit as st
import pandas as pd
from quant_system.utils.logger import get_logger

LOG = get_logger("replay_control_bar")


class ReplayControlBar:

    def __init__(self, replay_engine, timeline, dashboard):
        self.engine = replay_engine
        self.timeline = timeline
        self.dashboard = dashboard

        if "replay_playing" not in st.session_state:
            st.session_state["replay_playing"] = False

        if "replay_speed" not in st.session_state:
            st.session_state["replay_speed"] = 1.0

        if "replay_seek" not in st.session_state:
            st.session_state["replay_seek"] = 0

        if "replay_focus_asset" not in st.session_state:
            st.session_state["replay_focus_asset"] = None

    # ------------------------------------------------------------------
    def render(self):
        with st.container():
            self._render_controls()
            self._render_timeline_slider()
            self._render_event_jumpers()
            self._render_asset_focus_selector()

    # ------------------------------------------------------------------
    def _render_controls(self):
        c1, c2, c3 = st.columns([1, 1, 2])

        if c1.button("▶ Play", use_container_width=True):
            st.session_state["replay_playing"] = True
            self.engine.running = True
            self.engine.set_speed(st.session_state["replay_speed"])
            LOG.info("[Replay] Play")

        if c1.button("⏸ Pause", use_container_width=True):
            st.session_state["replay_playing"] = False
            self.engine.running = False
            LOG.info("[Replay] Pause")

        if c2.button("⏪ Step Back", use_container_width=True):
            idx = max(0, st.session_state["replay_seek"] - 1)
            st.session_state["replay_seek"] = idx
            self.engine.seek(idx)
            self._broadcast_seek(idx)

        if c2.button("⏩ Step Forward", use_container_width=True):
            idx = min(self.timeline.length - 1, st.session_state["replay_seek"] + 1)
            st.session_state["replay_seek"] = idx
            self.engine.seek(idx)
            self._broadcast_seek(idx)

        speed = c3.slider(
            "Speed (sec/bar)",
            min_value=0.01,
            max_value=3.0,
            value=st.session_state["replay_speed"],
            step=0.01
        )
        if speed != st.session_state["replay_speed"]:
            st.session_state["replay_speed"] = speed
            self.engine.set_speed(speed)

    # ------------------------------------------------------------------
    def _render_timeline_slider(self):
        idx = st.slider(
            "Timeline",
            min_value=0,
            max_value=self.timeline.length - 1,
            value=st.session_state["replay_seek"],
            step=1
        )
        if idx != st.session_state["replay_seek"]:
            st.session_state["replay_seek"] = idx
            self.engine.seek(idx)
            self._broadcast_seek(idx)

    # ------------------------------------------------------------------
    def _render_event_jumpers(self):
        df = getattr(self.dashboard, "trade_log_df", None)
        if df is None or df.empty:
            return

        col1, col2, col3 = st.columns(3)

        if col1.button("Next Entry"):
            idx = self._find_next(df, "entry")
            if idx is not None:
                self._jump(idx)

        if col2.button("Next Exit"):
            idx = self._find_next(df, "exit")
            if idx is not None:
                self._jump(idx)

        if col3.button("Next A+"):
            idx = self._find_next(df[(df["tier"] == "A+")], "entry")
            if idx is not None:
                self._jump(idx)

    # ------------------------------------------------------------------
    def _render_asset_focus_selector(self):
        assets = list(self.engine.states.keys())
        selected = st.selectbox(
            "Asset Focus",
            ["ALL"] + assets,
            index=0
        )

        if selected != st.session_state["replay_focus_asset"]:
            st.session_state["replay_focus_asset"] = selected
            self._broadcast_asset_focus(selected)

    # ------------------------------------------------------------------
    def _jump(self, idx: int):
        st.session_state["replay_seek"] = idx
        self.engine.seek(idx)
        self._broadcast_seek(idx)

    # ------------------------------------------------------------------
    def _find_next(self, df, event_type: str):
        cursor_dt = self.timeline.dt_at(st.session_state["replay_seek"])
        candidates = df[df["dt"] > cursor_dt]
        if not candidates.empty:
            target_dt = candidates.iloc[0]["dt"]
            return self.timeline.timeline.index(target_dt)
        return None

    # ------------------------------------------------------------------
    def _broadcast_seek(self, idx: int):
        dt = self.timeline.dt_at(idx)
        script = f"""
        <script>
            window.postMessage({{
                type: "replay_seek",
                dt: "{dt}"
            }}, "*");
        </script>
        """
        st.markdown(script, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    def _broadcast_asset_focus(self, asset: str):
        script = f"""
        <script>
            window.postMessage({{
                type: "replay_focus_asset",
                asset: "{asset}"
            }}, "*");
        </script>
        """
        st.markdown(script, unsafe_allow_html=True)
