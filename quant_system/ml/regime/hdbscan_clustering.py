"""
HDBSCAN-based liquidity/chop clustering on 15m/1h features.

Given a timeframe CSV (e.g., XBTUSD_15m.csv or XBTUSD_1h.csv), this module:
 - builds a compact feature matrix from liquidity/volatility fields
 - standardizes features
 - fits HDBSCAN to find microstructure regimes (chop, anomaly, acceleration, etc.)
 - saves cluster assignments with outlier flag (-1 -> outlier)
 - reports cluster sizes and basic fit metadata.

Dependencies: hdbscan (with numpy/pandas/sklearn).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 15
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    features: Optional[List[str]] = None  # if None, auto-select common liquidity/vol features


class HDBSCANClusterer:
    def __init__(self, cfg: HDBSCANConfig):
        if not _HAS_HDBSCAN:
            raise ImportError("hdbscan is required for HDBSCANClusterer. Please install hdbscan.")
        self.cfg = cfg
        self.model_: Optional[hdbscan.HDBSCAN] = None
        self.selected_features_: List[str] = []
        self.report_: Dict[str, Any] = {}

    def _auto_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "volume",
            "dollar_volume",
            "atr",
            "range_pct",
            "realized_vol",
            "volatility_z",
            "spread_bps",
            "toxicity",
        ]
        return [c for c in candidates if c in df.columns]

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if self.cfg.features:
            feats = [c for c in self.cfg.features if c in df.columns]
        else:
            feats = self._auto_features(df)
        if not feats:
            raise ValueError("No usable features found for HDBSCAN clustering.")
        self.selected_features_ = feats
        X = df[feats].astype(float).copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0.0))
        return X_scaled

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._build_matrix(df)
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric=self.cfg.metric,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            cluster_selection_method=self.cfg.cluster_selection_method,
            allow_single_cluster=self.cfg.allow_single_cluster,
        )
        labels = model.fit_predict(X)
        self.model_ = model

        out_df = df[["dt"]].copy()
        out_df["cluster"] = labels
        out_df["is_outlier"] = (labels == -1).astype(int)

        # report
        vals, counts = np.unique(labels, return_counts=True)
        clusters = {int(v): int(c) for v, c in zip(vals, counts)}
        self.report_ = {
            "params": asdict(self.cfg),
            "features": self.selected_features_,
            "n_obs": int(len(df)),
            "clusters": clusters,
        }
        return out_df

    def save(self, out_dir: Path, asset: str, tf: str, clusters: pd.DataFrame):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{asset}_{tf}_clusters.csv"
        clusters.to_csv(out_csv, index=False)
        with open(out_dir / "train_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report_, f, indent=2)

