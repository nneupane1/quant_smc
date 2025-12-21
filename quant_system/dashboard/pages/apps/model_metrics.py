import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("model_metrics")


# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------
def _load_metrics():
    base = Path.cwd() / "ml_outputs"

    metrics_file = base / "metrics.json"
    shap_file = base / "shap_values.json"
    shap_mean_file = base / "shap_mean.json"
    trans_file = base / "regime_transition.json"
    hazard_file = base / "hazard_curve.json"
    quant_file = base / "quantile_eval.json"

    out = {}

    def read(path):
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    out["metrics"] = read(metrics_file)
    out["shap_values"] = read(shap_file)
    out["shap_mean"] = read(shap_mean_file)
    out["transition"] = read(trans_file)
    out["hazard"] = read(hazard_file)
    out["quantile"] = read(quant_file)
    return out


# ------------------------------------------------------------
# Visualization Helpers
# ------------------------------------------------------------
def _title(label):
    st.markdown(
        f"""
        <h2 style="margin-bottom:4px;">{label}</h2>
        <hr style="margin-top:6px;margin-bottom:16px;opacity:0.2;">
        """,
        unsafe_allow_html=True,
    )


def _metrics_table(metrics):
    st.dataframe(pd.DataFrame(metrics).T, height=250)


def _plot_pr_curve(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="recall:Q", y="precision:Q")
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _plot_roc_curve(df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="fpr:Q", y="tpr:Q")
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _calibration_plot(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="predicted:Q", y="observed:Q")
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _shap_bar(mean_importance):
    df = pd.DataFrame(mean_importance, columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=False).head(25)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="importance:Q",
            y=alt.Y("feature:N", sort="-x"),
            color=alt.value("#5db7ff"),
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _shap_waterfall(shap_values):
    df = pd.DataFrame(shap_values)
    df = df.sort_values("value", ascending=False).head(20)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="value:Q",
            y=alt.Y("feature:N", sort="-x"),
            color=alt.value("#f78c6b"),
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _transition_matrix_plot(trans):
    df = pd.DataFrame(trans)
    st.dataframe(df.style.background_gradient(cmap="Blues"), height=280)


def _hazard_curve_plot(h):
    df = pd.DataFrame(h)
    df["time"] = df["time"].astype(float)

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="time:Q", y="hazard:Q")
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def _quantile_coverage_plot(q):
    df = pd.DataFrame(q)

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="quantile:Q", y="coverage:Q")
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------
# RENDER PAGE
# ------------------------------------------------------------
def render_metrics(theme_choice, model_version):
    LOG.info("Rendering Model Metrics Dashboard")

    st.markdown(
        f"""
        <h1>Model Metrics</h1>
        <span style="color:#888;">ML Diagnostics • Calibration • SHAP • Hazard • Regime</span>
        <hr style="margin-top:12px;margin-bottom:20px;opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )

    artifacts = _load_metrics()
    metrics = artifacts["metrics"]

    if metrics is None:
        st.warning("No metrics found.")
        return

    # MAIN METRICS TABLE
    _title("Model Summary Metrics")
    _metrics_table(metrics)

    # SIDE-BY-SIDE CURVES
    col1, col2, col3 = st.columns(3)

    with col1:
        if "pr_curve" in metrics:
            _title("PR Curve")
            df = pd.DataFrame(metrics["pr_curve"])
            _plot_pr_curve(df)

    with col2:
        if "roc_curve" in metrics:
            _title("ROC Curve")
            df = pd.DataFrame(metrics["roc_curve"])
            _plot_roc_curve(df)

    with col3:
        if "calibration_curve" in metrics:
            _title("Calibration")
            df = pd.DataFrame(metrics["calibration_curve"])
            _calibration_plot(df)

    # SHAP IMPORTANCE
    if artifacts["shap_mean"]:
        _title("SHAP — Top Feature Importances")
        _shap_bar(artifacts["shap_mean"])

    if artifacts["shap_values"]:
        _title("SHAP — Waterfall")
        _shap_waterfall(artifacts["shap_values"])

    # REGIME TRANSITIONS
    if artifacts["transition"]:
        _title("Regime Transition Matrix (HMM)")
        _transition_matrix_plot(artifacts["transition"])

    # HAZARD MODEL
    if artifacts["hazard"]:
        _title("Hazard Model — Survival Curve")
        _hazard_curve_plot(artifacts["hazard"])

    # QUANTILE REGRESSION
    if artifacts["quantile"]:
        _title("Quantile Coverage")
        _quantile_coverage_plot(artifacts["quantile"])
