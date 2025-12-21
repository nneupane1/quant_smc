"""
portfolio_heatmap.py
Multi-asset portfolio exposure, risk-weight, correlation, and regime heatmap.

Displays:
 • Long/short exposure %
 • Net exposure %
 • Volatility-weighted position sizes
 • Correlation clusters (1h returns)
 • Regime state overlay (trend/range/expansion/collapse)

Everything updates dynamically without page refresh.
"""

import streamlit as st
import numpy as np
import pandas as pd
from quant_system.utils.logger import get_logger

LOG = get_logger("portfolio_heatmap")


class PortfolioHeatmapPanel:

    def __init__(self, engine, dashboard):
        """
        engine: ForwardEngine or ForwardWrapper (has exposure state)
        dashboard: dashboard adapter for JS postMessage
        """
        self.engine = engine
        self.dashboard = dashboard

        if "heatmap_init" not in st.session_state:
            st.session_state["heatmap_init"] = False

    # -----------------------------------------------------------------
    def render(self):
        self._inject_css_js()

        # container
        st.subheader("Portfolio Exposure Heatmap")
        placeholder = st.empty()

        # Build heatmap data
        exposures = self.engine.exposure_tracker.current_exposures()   # {asset:{long,short,net,...}}
        regimes = self.engine.regime_state  # {asset:"trend"/"range"/...}
        vols = self.engine.volatility_state # {asset:vol_zscore}

        corr = self._compute_corr()

        # final table for JS
        heat_values = self._build_heatgrid(exposures, regimes, vols, corr)

        payload = {
            "type": "portfolio_heatmap_update",
            "data": heat_values
        }

        # send to JS
        script = f"""
        <script>
            window.postMessage({payload}, "*");
        </script>
        """
        placeholder.markdown(script, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    def _compute_corr(self):
        """Compute rolling 1h correlation matrix across assets."""
        df = self.engine.market_state.get_1h_matrix()   # shape: (T, assets)
        if df is None or df.shape[0] < 50:
            return {}

        corr = df.pct_change().corr()
        return corr.fillna(0).round(2).to_dict()

    # -----------------------------------------------------------------
    def _build_heatgrid(self, exposures, regimes, vols, corr):
        out = []

        for asset in exposures:
            exp = exposures[asset]
            reg = regimes.get(asset, "unknown")
            vol = float(vols.get(asset, 0))

            row = {
                "asset": asset,
                "long": float(exp.get("long", 0)),
                "short": float(exp.get("short", 0)),
                "net": float(exp.get("net", 0)),
                "risk_weight": float(exp.get("risk_weight", 0)),
                "vol_z": vol,
                "regime": reg,
                "corr": corr.get(asset, {})
            }
            out.append(row)

        return out

    # -----------------------------------------------------------------
    def _inject_css_js(self):
        if st.session_state["heatmap_init"]:
            return

        with open("quant_system/dashboard/components/portfolio_heatmap/heatmap.css") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

        with open("quant_system/dashboard/components/portfolio_heatmap/heatmap.js") as f:
            js = f"<script>{f.read()}</script>"
            st.markdown(js, unsafe_allow_html=True)

        st.session_state["heatmap_init"] = True
