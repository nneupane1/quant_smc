from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.intelligence import execution_snapshot, grouped_features, latest_market_row, latest_reasoning
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def _preview(group: dict, limit: int = 16) -> pd.DataFrame:
    items = list(group.items())[:limit]
    return pd.DataFrame(items, columns=["feature", "value"])


def render_insights(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    row = latest_market_row(context)
    reasoning = latest_reasoning(context)
    groups = grouped_features(row)
    snap = execution_snapshot(context)
    gates = snap.get("gates", {})
    conf = snap.get("confluence", {})
    side_badge = status_badge(f"Side {snap.get('side', 'n/a')}", "neutral")

    page_header(
        "Insights",
        "A research-grade state inspection layer showing structure, liquidity, momentum, and execution constraints as one causal surface.",
        kicker="Intelligence Layer",
    )
    st.markdown(
        f"{status_badge('Eligibility pass' if gates.get('passed') else 'Eligibility constrained', 'good' if gates.get('passed') else 'warn')} "
        f"{side_badge}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Confluence", "value": f"{float(conf.get('confluence_score') or 0.0):.3f}"},
            {"label": "Hazard", "value": f"{float(snap.get('hazard') or 0.0):.3f}"},
            {"label": "EVR", "value": f"{float(snap.get('evr') or 0.0):.3f}"},
            {"label": "Median R", "value": f"{float(snap.get('median_r') or 0.0):.2f}"},
            {"label": "Trend Prob", "value": f"{float(row.get('p_regime_trend') or 0.0):.3f}"},
            {"label": "Flow Prob", "value": f"{float(row.get('p_flow_1h', row.get('prob_flow_1h') or 0.0) or 0.0):.3f}"},
        ]
    )

    col1, col2 = st.columns([1.15, 1.0])
    with col1:
        section_title("Execution Constraints", "Decision eligibility under 12h/6h/1h gates")
        st.json(gates)
        section_title("Causal Trace", "Latest deterministic reasoning payload")
        if reasoning:
            st.json(reasoning)
        else:
            st.info("No reasoning payload is available yet.")
    with col2:
        section_title("State Preview", "Market configuration exposed as feature groups")
        st.markdown(
            f"{status_badge('Structure', 'neutral')} {status_badge('Liquidity', 'neutral')} {status_badge('Momentum', 'neutral')} {status_badge('Risk', 'neutral')}",
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["Structure", "Liquidity Geometry", "Momentum & VWAP", "Risk Surface"])
    with tabs[0]:
        section_title("Structural Bias and Repair", "Retests, BOS/CHOCH, order-block and FVG state")
        df = _preview(groups["structure"])
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No structural features present.")
    with tabs[1]:
        section_title("Liquidity Geometry", "Pool distance, wick-flow asymmetry, sweep state")
        df = _preview(groups["liquidity"])
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No liquidity diagnostics present.")
    with tabs[2]:
        section_title("Momentum Curvature and Deformation", "Flow impulse, EMA geometry, VWAP-related features if present")
        df = _preview(groups["momentum"])
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No momentum diagnostics present.")
    with tabs[3]:
        section_title("Expectancy and Fragility", "Hazard, EVR, tail and execution risk context")
        df = _preview(groups["risk"])
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No risk diagnostics present.")
