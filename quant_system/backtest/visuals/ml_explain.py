import streamlit as st
import shap
import pandas as pd
from quant_system.ml.model_registry import ModelRegistry


def explain_trade(model_version: str, trade_features: pd.DataFrame):
    """
    SHAP explanation for a single trade's features.
    Only LightGBM/XGB supported.
    """
    registry = ModelRegistry()
    model = registry.load_model(model_version, "meta")  # meta model governs confluence

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(trade_features)

    st.subheader("ML Explainability (SHAP)")
    shap_html = shap.plots.force(explainer.expected_value, values, trade_features, matplotlib=False)
    st.components.v1.html(shap_html.html(), height=350)
