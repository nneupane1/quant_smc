import streamlit as st
from pathlib import Path
from quant_system.utils.logger import get_logger
from quant_system.config.config_loader import ConfigLoader

LOG = get_logger("dashboard_app")

# Page config
st.set_page_config(page_title="Quant System", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Theme manager (if present)
try:
    from quant_system.dashboard.theme_manager import ThemeManager
    theme_manager = ThemeManager()
    theme_choice = st.sidebar.selectbox("Theme", ["dark", "bloomberg", "minimal", "high_contrast"], index=0)
    theme_manager.apply_theme(theme_choice)
except Exception:
    theme_choice = "dark"

# Config loader
config_path = Path(__file__).parent.parent / "config"
loader = ConfigLoader(str(config_path))
cfg = loader.load()

# Sidebar Navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Live Trading",
        "Backtest Results",
        "Model Metrics",
        "Trade Journal",
        "Settings",
    ],
)

LOG.info(f"Dashboard initialized - Page={selected_page}")

# Page Router
if selected_page == "Home":
    st.title("Home")
    st.write("Welcome to Quant System Dashboard")
elif selected_page == "Live Trading":
    try:
        from quant_system.dashboard.pages.apps.live_monitor import render_live
        render_live(theme_choice)
    except Exception as e:
        st.error(f"Live monitor not available: {e}")
elif selected_page == "Backtest Results":
    try:
        from quant_system.dashboard.pages.apps.pnl_dashboard import render_backtest
        render_backtest(theme_choice)
    except Exception as e:
        st.error(f"Backtest page not available: {e}")
elif selected_page == "Model Metrics":
    try:
        from quant_system.dashboard.pages.apps.model_metrics import render_metrics
        render_metrics(theme_choice)
    except Exception as e:
        st.error(f"Model metrics not available: {e}")
elif selected_page == "Trade Journal":
    try:
        from quant_system.dashboard.pages.apps.trade_journal import render_journal
        render_journal(theme_choice)
    except Exception as e:
        st.error(f"Trade journal not available: {e}")
elif selected_page == "Settings":
    st.title("Settings")
    st.write("Edit system-level configurations dynamically here.")
