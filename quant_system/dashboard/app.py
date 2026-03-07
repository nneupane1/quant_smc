from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.theme_manager import ThemeManager
from quant_system.dashboard.ui import inject_page_notice, page_header, section_title, status_badge
from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter
from quant_system.utils.logger import get_logger

LOG = get_logger("dashboard_app")
LOGO_PATH = Path(__file__).resolve().parent / "logo" / "bull_bear.png"

st.set_page_config(
    page_title="Quant System Terminal",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dashboard_adapter" not in st.session_state:
    st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()

theme_manager = ThemeManager()

PAGE_REGISTRY = {
    "Mission Control": ("quant_system.dashboard.pages.apps.mission_control", "render_mission_control"),
    "Insights": ("quant_system.dashboard.pages.apps.insights", "render_insights"),
    "Regime Briefings": ("quant_system.dashboard.pages.apps.regime_briefings", "render_regime_briefings"),
    "Signal Intelligence": ("quant_system.dashboard.pages.apps.signal_intelligence", "render_signal_intelligence"),
    "Decision Trace": ("quant_system.dashboard.pages.apps.decision_trace", "render_decision_trace"),
    "Performance Intelligence": ("quant_system.dashboard.pages.apps.trade_log", "render_trade_log"),
    "Risk Radar": ("quant_system.dashboard.pages.apps.risk_radar", "render_risk_radar"),
    "Research & Audit": ("quant_system.dashboard.pages.apps.research_audit", "render_research_audit"),
    "Settings": (None, None),
}

LEGACY_REGISTRY = {
    "Home": ("quant_system.dashboard.pages.apps.pnl_dashboard", "render_home"),
    "Live Trading": ("quant_system.dashboard.pages.apps.live_monitor", "render_live"),
    "Forward Test": ("quant_system.dashboard.pages.apps.forward_test", "render_forward_test"),
    "Backtest Results": ("quant_system.dashboard.pages.apps.pnl_dashboard", "render_backtest"),
    "Replay Mode": ("quant_system.dashboard.pages.apps.replay_mode", "render_replay_mode"),
    "Replay Timeline": ("quant_system.dashboard.pages.apps.replay_timeline", "render_replay_timeline"),
    "Risk Attribution": ("quant_system.dashboard.pages.apps.risk_attribution", "render_risk_attribution"),
    "SMC Inspector": ("quant_system.dashboard.pages.apps.smc_inspector", "render_smc_inspector"),
    "Model Metrics": ("quant_system.dashboard.pages.apps.model_metrics", "render_metrics"),
    "Trade Journal": ("quant_system.dashboard.pages.apps.trade_journal", "render_journal"),
    "Trade Log": ("quant_system.dashboard.pages.apps.trade_log", "render_trade_log"),
}

DOMAIN_CAPTIONS = {
    "Mission Control": "Account state, active trades, execution tape, capital cycle",
    "Insights": "Causal trace, feature graph, structural state, execution eligibility",
    "Regime Briefings": "12h state, persistence, transition risk, structural stability",
    "Signal Intelligence": "Comparative ranking, confluence vectors, candidate coherence",
    "Decision Trace": "Per-alert causal chain, entry/exit rationale, lifecycle reconstruction",
    "Performance Intelligence": "Per-trade ledger, PnL decomposition, win-rate and ML attribution surfaces",
    "Risk Radar": "Stress gates, fragility, liquidity degradation, readiness control",
    "Research & Audit": "Backtest, replay, trade ledger, reasoning reconstruction",
}


def _set_active_page(page_name: str) -> None:
    st.session_state["dashboard_active_page"] = page_name


def _active_page() -> str:
    return st.session_state.get("dashboard_active_page", "Mission Control")


def _render_settings(context):
    page_header(
        "Terminal Settings",
        "Control the active artifact directories and shell theme without changing code paths.",
        kicker="Environment",
    )
    inject_page_notice(
        "The dashboard remains Streamlit-native so it stays connected to the repaired adapters and CLI outputs."
    )
    st.json(
        {
            "backtest_dir": str(context.backtest_dir),
            "forward_dir": str(context.forward_dir),
            "model_dir": str(context.model_dir),
            "theme": context.theme_choice,
            "model_version": context.model_version,
        }
    )


def _render_domain_dock() -> None:
    section_title("Desk", "Primary operator domains")
    labels = list(PAGE_REGISTRY.keys())
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            if st.button(label, key=f"dock_{label}", use_container_width=True):
                _set_active_page(label)
            st.markdown(
                f"<div class='qs-nav-caption'>{DOMAIN_CAPTIONS.get(label, 'System settings')}</div>",
                unsafe_allow_html=True,
            )


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    st.title("Quant Terminal")
    theme_choice = st.selectbox("Theme", ["bloomberg", "dark", "minimal", "high_contrast"], index=0)
    theme_manager.apply_theme(theme_choice)

    backtest_dir = st.text_input("Backtest Dir", value="backtest_outputs")
    forward_dir = st.text_input("Forward Dir", value="forward_outputs")
    model_dir = st.text_input("Model Dir", value="models")
    telemetry_url = st.text_input(
        "Telemetry API",
        value=os.environ.get("QUANT_TERMINAL_API_BASE", "http://127.0.0.1:8100"),
    )
    telemetry_refresh = st.number_input("Telemetry Refresh (s)", min_value=1, max_value=60, value=5, step=1)

    st.markdown("### Domains")
    for label in PAGE_REGISTRY.keys():
        if st.button(label, key=f"sidebar_domain_{label}", use_container_width=True):
            _set_active_page(label)

    with st.expander("Detailed Views", expanded=False):
        for label in LEGACY_REGISTRY.keys():
            if st.button(label, key=f"sidebar_legacy_{label}", use_container_width=True):
                _set_active_page(label)

context = build_context(
    theme_choice,
    backtest_dir=backtest_dir,
    forward_dir=forward_dir,
    model_dir=model_dir,
    adapter=st.session_state["dashboard_adapter"],
    telemetry_url=telemetry_url,
)
st.session_state["dashboard_context"] = context

selected_page = _active_page()

LOG.info("Dashboard initialized page=%s theme=%s", selected_page, theme_choice)

if LOGO_PATH.exists():
    logo_col, _ = st.columns([1.2, 8.8])
    with logo_col:
        st.image(str(LOGO_PATH), width=140)

status_col1, status_col2, status_col3 = st.columns([1.3, 1.2, 2.5])
with status_col1:
    tone = "good" if context.transport == "telemetry_api" else "neutral"
    st.markdown(status_badge(f"{theme_choice} via {context.transport}", tone), unsafe_allow_html=True)
with status_col2:
    tone = "good" if context.model_version != "unavailable" else "warn"
    st.markdown(status_badge(f"Models {context.model_version}", tone), unsafe_allow_html=True)
with status_col3:
    bt_count = len(context.backtest["trades"])
    st.markdown(
        status_badge(f"Backtest trades {bt_count}", "good" if bt_count else "warn"),
        unsafe_allow_html=True,
    )

_render_domain_dock()

registry = PAGE_REGISTRY if selected_page in PAGE_REGISTRY else LEGACY_REGISTRY

module_path, fn_name = registry[selected_page]
if selected_page == "Settings":
    _render_settings(context)
else:
    try:
        module = __import__(module_path, fromlist=[fn_name])
        getattr(module, fn_name)(theme_choice, context.model_version, context=context)
    except Exception as exc:
        LOG.exception("Dashboard page failed: %s", selected_page)
        st.error(f"{selected_page} is not available: {exc}")

if context.transport == "telemetry_api":
    components.html(
        f"""
        <script>
          window.setTimeout(function() {{
            window.parent.location.reload();
          }}, {int(telemetry_refresh) * 1000});
        </script>
        """,
        height=0,
    )
