from typing import Optional
from pathlib import Path

import streamlit as st

from quant_system.utils.logger import get_logger

LOG = get_logger("theme_manager")

THEME_CSS_CACHE_KEY = "active_theme_css"


class ThemeManager:
    """Manages hot-swappable CSS themes for the Streamlit dashboard shell."""

    def __init__(self, assets_dir: Optional[Path] = None):
        self.assets_dir = Path(assets_dir) if assets_dir is not None else Path(__file__).parent
        self.theme_dir = self.assets_dir / "styles"
        self.global_css_path = self.theme_dir / "global.css"

    def load_css(self, theme_name: str) -> str:
        css_path = self.theme_dir / f"{theme_name}.css"
        if not css_path.exists() and theme_name == "high_contrast":
            css_path = self.theme_dir / "highh_contrast.css"
        if not css_path.exists():
            LOG.warning(f"Theme CSS not found: {css_path}")
            return ""
        return css_path.read_text(encoding="utf-8")

    def _load_global(self) -> str:
        if not self.global_css_path.exists():
            return ""
        return self.global_css_path.read_text(encoding="utf-8")

    def apply_theme(self, theme_name: str):
        css = self._load_global() + "\n" + self.load_css(theme_name)

        # Remove old theme
        st.session_state[THEME_CSS_CACHE_KEY] = css

        styled = f"<style id='dynamic-theme'>{css}</style>"
        st.markdown(styled, unsafe_allow_html=True)

        LOG.info(f"Applied theme: {theme_name}")

    def inject_current_theme(self):
        css = st.session_state.get(THEME_CSS_CACHE_KEY)
        if css:
            st.markdown(f"<style id='dynamic-theme'>{css}</style>", unsafe_allow_html=True)
