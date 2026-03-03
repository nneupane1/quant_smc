from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.intelligence import candidate_frame
from quant_system.dashboard.ui import metric_grid, page_header, section_title


def render_signal_intelligence(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    candidates = candidate_frame(context)

    page_header(
        "Signal Intelligence",
        "Comparative ranking of trade candidates using confluence, regime alignment, expectancy, and fragility in one reasoning vector.",
        kicker="Ranking Surface",
    )
    if candidates.empty:
        st.info("No candidate events or trade records are available to rank.")
        return

    metric_grid(
        [
            {"label": "Candidates", "value": f"{len(candidates)}"},
            {"label": "Top Score", "value": f"{float(candidates['signal_score'].max()):.3f}"},
            {"label": "Mean Confluence", "value": f"{float(candidates['confluence'].fillna(0.0).mean()):.3f}"},
            {"label": "Mean EVR", "value": f"{float(candidates['evr'].fillna(0.0).mean()):.3f}"},
        ]
    )

    section_title("Top Ranked Candidates", "Higher score means more coherent structure-probability-constraint alignment")
    st.dataframe(
        candidates[
            ["timestamp", "asset", "tier", "confluence", "evr", "median_r", "flow_1h", "hazard", "regime_state", "signal_score", "source"]
        ].head(25),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = candidates.head(25).copy()
    section_title("Coherence Map", "Confluence against expectancy, colored by fragility")
    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=110, opacity=0.8)
        .encode(
            x="confluence:Q",
            y="evr:Q",
            size="signal_score:Q",
            color=alt.Color("hazard:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["asset", "tier", "confluence", "evr", "median_r", "flow_1h", "hazard", "signal_score"],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
