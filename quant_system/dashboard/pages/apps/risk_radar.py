from __future__ import annotations

import altair as alt
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.intelligence import execution_snapshot, risk_surface
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def render_risk_radar(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    surface = risk_surface(context)
    state = context.forward["state"]
    exec_snap = execution_snapshot(context)

    page_header(
        "Risk Radar",
        "Forward-looking control surface for market stress, execution constraints, and system-readiness gates.",
        kicker="Control Surface",
    )
    st.markdown(
        f"{status_badge('Cooling engaged' if state.get('cooling_to') else 'Deployment allowed', 'warn' if state.get('cooling_to') else 'good')} "
        f"{status_badge('Gate pass' if exec_snap.get('gates', {}).get('passed') else 'Gate constrained', 'warn' if not exec_snap.get('gates', {}).get('passed') else 'good')}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Hazard", "value": f"{float(exec_snap.get('hazard') or 0.0):.3f}"},
            {"label": "Max Drawdown", "value": f"{float(state.get('max_drawdown') or 0.0):.3f}"},
            {"label": "Cooling", "value": str(bool(state.get('cooling_to')))},
            {"label": "Free Capital", "value": f"${float(state.get('free_capital') or 0.0):,.2f}"},
        ]
    )

    section_title("Risk Surface", "Higher bars indicate stress or constraint intensity")
    chart = (
        alt.Chart(surface)
        .mark_bar(color="#ff6b6b")
        .encode(x="value:Q", y=alt.Y("metric:N", sort="-x"), tooltip=["metric", "value", "description"])
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section_title("Gate Breakdown", "12h ocean, 6h waves, 1h flow")
        st.json(exec_snap.get("gates", {}))
    with col2:
        section_title("Execution State", "Capital and hedge posture")
        st.json(
            {
                "equity": state.get("equity"),
                "free_capital": state.get("free_capital"),
                "locked_profit": state.get("locked_profit"),
                "risk_mode": state.get("risk_mode"),
                "hedge_ratio": state.get("hedge_ratio"),
                "cooling_to": state.get("cooling_to"),
            }
        )
