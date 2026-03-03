from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import inject_page_notice, metric_grid, page_header, section_title


def _read_optional_json(path: Path):
    if not path.exists():
        return None
    import json

    with path.open("r") as handle:
        return json.load(handle)


def _optional_analysis(context: DashboardContext) -> dict:
    base = Path.cwd() / "ml_outputs"
    return {
        "shap_values": _read_optional_json(base / "shap_values.json"),
        "shap_mean": _read_optional_json(base / "shap_mean.json"),
        "transition": _read_optional_json(base / "regime_transition.json"),
        "hazard": _read_optional_json(base / "hazard_curve.json"),
        "quantile": _read_optional_json(base / "quantile_eval.json"),
    }


def _bar(df: pd.DataFrame, x: str, y: str, color: str = "#ffb000", height: int = 280):
    chart = (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(x=alt.X(x), y=alt.Y(y, sort="-x"), tooltip=list(df.columns))
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_metrics(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    summary = context.model_summary.copy()

    page_header(
        "Model Metrics",
        "Schema-driven registry view for the active model suite. The dashboard only trusts saved artifact metadata and metrics.",
        kicker="ML Registry",
    )

    if summary.empty:
        inject_page_notice("No model artifacts were found under the configured model registry directory.")
        return

    metric_grid(
        [
            {"label": "Models", "value": f"{len(summary)}"},
            {"label": "Latest Version", "value": model_version},
            {"label": "Avg Feature Count", "value": f"{summary['feature_count'].dropna().mean():.1f}"},
            {"label": "Best AUC", "value": f"{summary['auc'].dropna().max():.3f}" if summary['auc'].notna().any() else "--"},
        ]
    )

    section_title("Latest Model Registry", "One row per latest saved model artifact")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if summary["feature_count"].notna().any():
        section_title("Feature Width by Model", "Confirms per-model schema ownership")
        _bar(summary.dropna(subset=["feature_count"]), "feature_count:Q", "model:N")

    if summary["auc"].notna().any():
        section_title("AUC by Model", "Specialists, meta, confluence, and auxiliaries")
        auc_df = summary.dropna(subset=["auc"]).sort_values("auc", ascending=False)
        _bar(auc_df, "auc:Q", "model:N", color="#6ea8fe")

    analysis = _optional_analysis(context)
    col1, col2 = st.columns(2)
    with col1:
        if analysis["shap_mean"]:
            section_title("Top SHAP Features", "Optional flat analysis output")
            shap_df = pd.DataFrame(analysis["shap_mean"], columns=["feature", "importance"])
            _bar(shap_df.sort_values("importance", ascending=False).head(20), "importance:Q", "feature:N", color="#ff8f00")
    with col2:
        if analysis["quantile"]:
            section_title("Quantile Coverage", "Return forecaster calibration")
            q_df = pd.DataFrame(analysis["quantile"])
            chart = (
                alt.Chart(q_df)
                .mark_line(point=True, color="#3ddc97")
                .encode(x="quantile:Q", y="coverage:Q", tooltip=list(q_df.columns))
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

    if analysis["transition"]:
        section_title("Regime Transition Matrix", "Optional HMM transition diagnostics")
        st.dataframe(pd.DataFrame(analysis["transition"]), use_container_width=True)

    if analysis["hazard"]:
        section_title("Hazard Curve", "Optional survival diagnostics")
        hazard_df = pd.DataFrame(analysis["hazard"])
        chart = (
            alt.Chart(hazard_df)
            .mark_line(color="#ff6b6b")
            .encode(x="time:Q", y="hazard:Q", tooltip=list(hazard_df.columns))
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
