from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.intelligence import latest_market_row, recent_regime_history
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def render_regime_briefings(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    row = latest_market_row(context)
    history = recent_regime_history(context)
    current_state = row.get("regime_state", "unknown")
    trend = float(row.get("p_regime_trend") or 0.0)
    range_p = float(row.get("p_regime_range") or 0.0)
    exp = float(row.get("p_regime_expansion") or 0.0)
    coll = float(row.get("p_regime_collapse") or 0.0)
    transition_risk = 1.0 - max(trend, range_p, exp, coll)

    page_header(
        "Regime Briefings",
        "12h volatility-liquidity regime state, persistence, transition risk, and structural stability rendered as an active operating constraint.",
        kicker="State Briefing",
    )
    st.markdown(
        f"{status_badge(f'State {current_state}', 'neutral')} "
        f"{status_badge(f'Transition risk {transition_risk:.2f}', 'warn' if transition_risk > 0.35 else 'good')}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Trend", "value": f"{trend:.3f}"},
            {"label": "Range", "value": f"{range_p:.3f}"},
            {"label": "Expansion", "value": f"{exp:.3f}"},
            {"label": "Collapse", "value": f"{coll:.3f}"},
            {"label": "Toxicity 12h", "value": f"{float(row.get('toxicity_12h') or 0.0):.3f}"},
            {"label": "Zone Score 6h", "value": f"{float(row.get('zone_score_6h') or 0.0):.3f}"},
        ]
    )

    if history.empty:
        st.info("No regime history columns are available in the active market frame.")
        return

    section_title("Regime State Trajectory", "Recent 12h regime probabilities projected onto the execution spine")
    melt = history.melt(id_vars=["timestamp"], value_vars=[c for c in history.columns if c.startswith("p_regime_")], var_name="regime", value_name="probability")
    chart = (
        alt.Chart(melt)
        .mark_line(strokeWidth=2)
        .encode(x="timestamp:T", y="probability:Q", color="regime:N", tooltip=["timestamp:T", "regime:N", "probability:Q"])
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section_title("Persistence Context", "Dominant state confidence over time")
        pers = history.copy()
        regime_cols = [c for c in pers.columns if c.startswith("p_regime_")]
        pers["dominant_prob"] = pers[regime_cols].max(axis=1)
        pers["transition_risk"] = 1.0 - pers["dominant_prob"]
        st.dataframe(pers[["timestamp", "regime_state", "dominant_prob", "transition_risk"]].tail(30), use_container_width=True, hide_index=True)
    with col2:
        section_title("Structural Stability", "Compression, toxicity, and 6h zone support")
        stability_cols = [c for c in ["timestamp", "compression_12h", "toxicity_12h", "zone_score_6h"] if c in history.columns]
        st.dataframe(history[stability_cols].tail(30), use_container_width=True, hide_index=True) if len(stability_cols) > 1 else st.info("No additional stability columns available.")
