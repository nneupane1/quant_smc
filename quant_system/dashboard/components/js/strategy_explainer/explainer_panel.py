"""
explainer_panel.py
Full hierarchical strategy explanation panel:
 • SMC ⇨ Flow ⇨ ML ⇨ Confluence ⇨ EVR ⇨ Hazard ⇨ MPC
 • Uses JS for dynamic expand/collapse with no page reload.
"""

import json
import streamlit as st
from quant_system.utils.logger import get_logger

LOG = get_logger("strategy_explainer")


class StrategyExplainerPanel:

    def __init__(self):
        if "explainer_injected" not in st.session_state:
            st.session_state["explainer_injected"] = False

    # ---------------------------------------------------------------------
    def render(self, reasoning: dict):
        """
        reasoning = {
            "asset": "BTCUSDT",
            "timestamp": "...",
            "side": "long",
            "smc": {...},
            "flow": {...},
            "ml": {...},
            "confluence": {...},
            "evr": {...},
            "hazard": {...},
            "mpc": {...}
        }
        """
        self._inject_assets()

        st.subheader("Trade Reasoning Breakdown")

        payload = {
            "type": "strategy_explainer_update",
            "tree": reasoning
        }

        # Send to JS for rendering
        script = f"""
        <script>
            window.postMessage({json.dumps(payload)}, "*");
        </script>
        """
        st.markdown(script, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    def _inject_assets(self):
        if st.session_state["explainer_injected"]:
            return

        with open("quant_system/dashboard/components/strategy_explainer/explainer.css") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

        with open("quant_system/dashboard/components/strategy_explainer/explainer.js") as f:
            js = f"<script>{f.read()}</script>"
            st.markdown(js, unsafe_allow_html=True)

        st.session_state["explainer_injected"] = True
