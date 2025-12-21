"""
asset_selector.py
Streamlit-based multi-asset selector panel with instant front-end integration.

This panel:
 • draws a horizontal asset bar (BTC, ETH, SOL, AVAX, BNB)
 • highlights active selection
 • sends selection events to JS widgets (tv_chart, pnl_curve, panels)
 • notifies ForwardEngine through dashboard_adapter.set_active_asset()
"""

import streamlit as st
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("asset_selector")


class AssetSelector:
    """Bloomberg-style asset switching panel."""

    def __init__(self, cfg: ConfigLoader, dashboard_adapter=None):
        self.cfg = cfg
        self.dashboard = dashboard_adapter
        assets_cfg = cfg.load_yaml("assets.yaml")
        self.assets = assets_cfg["assets"]["enabled"]
        self.default = assets_cfg["default_asset"]

        # persistent state
        if "active_asset" not in st.session_state:
            st.session_state["active_asset"] = self.default

    # ------------------------------------------------------
    # RENDER PANEL
    # ------------------------------------------------------
    def render(self):
        st.markdown(
            """
            <style>
            .asset-btn-container {
                display: flex;
                gap: 12px;
                margin-bottom: 10px;
            }
            .asset-btn {
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 15px;
                cursor: pointer;
                background: #1a1a1a;
                border: 1px solid #333;
                transition: all 0.2s ease;
                color: #d0d0d0;
            }
            .asset-btn:hover {
                background: #333;
                border-color: #666;
            }
            .asset-btn-active {
                background: #0059ff;
                color: white;
                border: 1px solid #1b5cff;
                box-shadow: 0 0 10px #0059ffaa;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="asset-btn-container">', unsafe_allow_html=True)

        cols = st.columns(len(self.assets))
        for idx, asset in enumerate(self.assets):
            with cols[idx]:
                active = (asset == st.session_state["active_asset"])
                css_class = "asset-btn asset-btn-active" if active else "asset-btn"

                if st.button(
                    asset,
                    key=f"asset_{asset}",
                    help=f"Switch to {asset}",
                    use_container_width=True
                ):
                    st.session_state["active_asset"] = asset
                    LOG.info(f"[Dashboard] Asset switched to {asset}")

                    # Notify backend engine
                    if self.dashboard:
                        self.dashboard.set_active_asset(asset)

                    # Broadcast to JS widgets (TV chart, panels)
                    self._notify_js(asset)

                st.markdown(
                    f'<div class="{css_class}" style="text-align:center">{asset}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # BROADCAST TO JS WIDGETS
    # ------------------------------------------------------
    def _notify_js(self, asset: str):
        """Send postMessage to all frontend widgets."""
        script = f"""
            <script>
                window.postMessage({{
                    type: "asset_switch",
                    asset: "{asset}"
                }}, "*");
            </script>
        """
        st.markdown(script, unsafe_allow_html=True)
        LOG.info(f"[Dashboard] JS notified for asset={asset}")
