import streamlit as st
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("theme_manager")

THEME_CSS_CACHE_KEY = "active_theme_css"


class ThemeManager:
    """
    Manages hot-swappable CSS themes at runtime without page refresh.
    """

    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.theme_dir = assets_dir / "themes"

    def load_css(self, theme_name: str) -> str:
        css_path = self.theme_dir / f"{theme_name}.css"
        if not css_path.exists():
            LOG.warning(f"Theme CSS not found: {css_path}")
            return ""
        return css_path.read_text()

    def apply_theme(self, theme_name: str):
        css = self.load_css(theme_name)

        # Remove old theme
        st.session_state[THEME_CSS_CACHE_KEY] = css

        styled = f"<style id='dynamic-theme'>{css}</style>"
        st.markdown(styled, unsafe_allow_html=True)

        LOG.info(f"Applied theme: {theme_name}")

    def inject_current_theme(self):
        css = st.session_state.get(THEME_CSS_CACHE_KEY)
        if css:
            st.markdown(f"<style id='dynamic-theme'>{css}</style>", unsafe_allow_html=True)
