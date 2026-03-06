"""
Shared CLI helpers for building/loading feature frames and saving artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.label_builder import LabelBuilder
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.training.feature_builder import FeatureBuilder


def _quant_root(module_file: str) -> Path:
    path = Path(module_file).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "config").is_dir() and (parent / "__init__.py").exists():
            return parent
    raise FileNotFoundError(f"Could not resolve quant_system root from {module_file}")


def default_conf_dir(module_file: str) -> str:
    return str(_quant_root(module_file) / "config")


def default_dashboard_path(module_file: str) -> str:
    return str(_quant_root(module_file) / "dashboard" / "app.py")


def resolve_conf_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def load_registry(cfg: Dict[str, Any], override: Optional[str] = None) -> ModelRegistry:
    registry_path = (
        override
        or cfg.get("paths", {}).get("model_registry")
        or cfg.get("models", {}).get("registry_path")
        or "models"
    )
    return ModelRegistry(registry_path)


def default_asset(cfg: Dict[str, Any], override: Optional[str] = None) -> str:
    if override:
        return override
    return (
        cfg.get("default_asset")
        or cfg.get("assets", {}).get("default_asset")
        or "XBTUSD"
    )


def read_frame(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path_obj, nrows=0)
    # Avoid parsing numeric epoch columns as datetime strings (noise + slow path).
    parse_dates = [c for c in ("dt", "entry_ts", "exit_ts") if c in header.columns]
    df = pd.read_csv(
        path_obj,
        parse_dates=parse_dates if parse_dates else None,
        low_memory=False,
    )
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["dt"] = pd.to_datetime(ts, unit="s", errors="coerce")
        else:
            df["dt"] = pd.to_datetime(ts, errors="coerce")
    return df


def load_or_build_features(
    loader: ConfigLoader,
    *,
    asset: str,
    features_csv: Optional[str] = None,
    tf_dir: Optional[str] = None,
    features_out: Optional[str] = None,
) -> pd.DataFrame:
    if features_csv:
        return read_frame(features_csv)
    if not tf_dir:
        raise ValueError("Provide either --features or --tf-dir.")

    builder = FeatureBuilder(loader)
    df = builder.build_from_dir(tf_dir, asset, out_path=features_out)
    if features_out:
        out_path = Path(features_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        if builder.audit_table is not None and not builder.audit_table.empty:
            builder.audit_table.to_csv(out_path.parent / f"{asset}_htf_audit.csv", index=False)
    return df


def load_or_build_labels(
    loader: ConfigLoader,
    *,
    features_df: pd.DataFrame,
    labels_csv: Optional[str] = None,
    labels_out: Optional[str] = None,
) -> pd.DataFrame:
    if labels_csv:
        return read_frame(labels_csv)
    labeler = LabelBuilder(loader)
    df = labeler.apply(features_df)
    if labels_out:
        out_path = Path(labels_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def to_jsonable(obj: Any):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(record) for record in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {k: to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def save_json(path: str | Path, payload: Any):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
