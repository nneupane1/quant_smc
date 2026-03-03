from typing import Any, Dict, Optional

import pandas as pd

from quant_system.ml.registry.model_registry import ModelRegistry


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def explain_trade(
    trade_features: pd.DataFrame,
    model_name: str = "confluence_model",
    registry_dir: str = "models",
    render: bool = True,
) -> Dict[str, Any]:
    """
    Lightweight model introspection for a single trade row.
    Uses the saved model bundle and feature column list from the registry.
    """
    registry = ModelRegistry(registry_dir)
    clf, _cal, cfg = registry.load_latest_bundle(model_name)
    feature_cols = cfg.get("features", list(trade_features.columns))
    X = trade_features.reindex(columns=feature_cols).copy()

    prediction = None
    try:
        prediction = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    except Exception:
        prediction = None

    summary = {
        "model_name": model_name,
        "prediction": prediction,
        "features_used": feature_cols,
        "feature_values": X.iloc[0].dropna().to_dict() if not X.empty else {},
        "estimator": type(clf).__name__,
    }

    if not render:
        return summary

    st = _optional_streamlit()
    if st is None:
        return summary

    st.subheader("ML Explanation Snapshot")
    st.json(summary)
    return summary
