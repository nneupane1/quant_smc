"""
Unsupervised liquidity/microstructure clustering via HDBSCAN.

Designed for 15m / 1h bar CSVs (dt, open, high, low, close, volume, optional dollar_volume).
Computes a compact feature set (returns, range%, volume z-scores) → StandardScaler → HDBSCAN.
Saves model + scaler + meta; can emit label CSV (dt, cluster, outlier_flag).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover - surfaced at runtime
    raise ImportError("Please install hdbscan to use the clustering trainer.") from e


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    tf: str = "15m"  # for metadata only
    feature_window: int = 64  # rolling window for z-scores


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_features(df: pd.DataFrame, cfg: HDBSCANConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a compact, leak-safe feature matrix for clustering.
    """
    df = df.copy()
    df = df.sort_values("dt").reset_index(drop=True)

    # Basic required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core signals
    df["logret"] = np.log(df["close"]).diff()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["dollar_volume"] = df.get("dollar_volume", df["close"] * df["volume"])

    # Rolling z-scores (leak-safe: uses past only)
    w = max(cfg.feature_window, 8)
    for col in ["volume", "dollar_volume"]:
        if col in df.columns:
            mean = df[col].rolling(w, min_periods=8).mean()
            std = df[col].rolling(w, min_periods=8).std().replace(0, np.nan)
            df[f"{col}_z"] = (df[col] - mean) / std

    feat_cols = _safe_cols(df, ["logret", "range_pct", "volume_z", "dollar_volume_z"])
    feat_df = df[feat_cols].dropna()
    return feat_df, feat_cols


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class HDBSCANClusterer:
    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[hdbscan.HDBSCAN] = None
        self.features_: List[str] = []

    def fit(self, df_features: pd.DataFrame) -> Dict:
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df_features.values)

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric=self.cfg.metric,
        )
        labels = self.model.fit_predict(X)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_stats = {int(k): int(v) for k, v in zip(unique, counts)}

        self.features_ = list(df_features.columns)
        return {
            "n_clusters": int(len([c for c in unique if c != -1])),
            "cluster_counts": cluster_stats,
            "outlier_count": int(cluster_stats.get(-1, 0)),
        }

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model not fitted.")
        X = self.scaler.transform(df_features[self.features_].values)
        return self.model.fit_predict(X)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, out_dir: Path, meta_extra: Dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.model is None or self.scaler is None:
            raise RuntimeError("Fit the model before saving.")
        joblib.dump(self.model, out_dir / "model.joblib")
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
        meta = {
            "config": asdict(self.cfg),
            "features": self.features_,
        }
        meta.update(meta_extra)
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #
def run_hdbscan_training(
    csv_path: str,
    out_dir: str,
    cfg: Optional[HDBSCANConfig] = None,
    save_labels: bool = False,
) -> Dict:
    cfg = cfg or HDBSCANConfig()
    df = pd.read_csv(csv_path, parse_dates=["dt"])
    feat_df, feat_cols = build_features(df, cfg)

    trainer = HDBSCANClusterer(cfg)
    stats = trainer.fit(feat_df)
    trainer.save(Path(out_dir), {"stats": stats})

    if save_labels:
        labels = trainer.model.labels_  # type: ignore
        lab_df = pd.DataFrame(
            {"dt": df.loc[feat_df.index, "dt"].values, "cluster": labels}
        )
        lab_df["is_outlier"] = (lab_df["cluster"] == -1).astype(int)
        lab_df.to_csv(Path(out_dir) / "labels.csv", index=False)

    return stats

