"""
Unsupervised liquidity/chop clustering using HDBSCAN (fallback to DBSCAN).

Builds a feature matrix from bar data (15m/1h) with liquidity/vol/dispersion
signals, standardizes it, and fits an HDBSCAN model. Saves model, scaler,
and metadata; can also emit per-bar cluster assignments.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:  # optional dependency
    import hdbscan  # type: ignore

    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False


@dataclass
class LiquidityHDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "leaf"
    allow_single_cluster: bool = True
    seed: int = 42


class LiquidityClusterer:
    def __init__(self, cfg: LiquidityHDBSCANConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: List[str] = []
        self.meta: Dict = {}

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        f = f.sort_values("dt").reset_index(drop=True)

        # Required base columns
        for col in ("open", "high", "low", "close"):
            if col not in f.columns:
                f[col] = np.nan
            f[col] = pd.to_numeric(f[col], errors="coerce")
        if "volume" not in f.columns:
            f["volume"] = np.nan
        f["volume"] = pd.to_numeric(f["volume"], errors="coerce")

        # Derived
        logret = np.log(f["close"]).diff().fillna(0.0)
        f["abs_ret"] = logret.abs()
        f["range_pct"] = (f["high"] - f["low"]) / f["close"].replace(0, np.nan)
        f["dollar_volume"] = (
            f["dollar_volume"]
            if "dollar_volume" in f.columns
            else f["close"] * f["volume"]
        )
        f["wick_upper"] = (f["high"] - f["close"]) / f["close"].replace(0, np.nan)
        f["wick_lower"] = (f["close"] - f["low"]) / f["close"].replace(0, np.nan)

        # Realized volatility windows
        for win in (16, 64):
            f[f"rv_{win}"] = logret.rolling(win, min_periods=win // 2).std()

        feat_cols = [
            c
            for c in [
                "range_pct",
                "dollar_volume",
                "volume",
                "abs_ret",
                "rv_16",
                "rv_64",
                "wick_upper",
                "wick_lower",
                "spread",
            ]
            if c in f.columns
        ]
        feats = f[feat_cols].astype(float)
        feats = feats.dropna()
        self.feature_names = feat_cols
        return feats

    def fit(self, df: pd.DataFrame) -> Dict:
        feats = self._make_features(df)
        X = self.scaler.fit_transform(feats.values)

        if _HAS_HDBSCAN:
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.cfg.min_cluster_size,
                min_samples=self.cfg.min_samples,
                metric=self.cfg.metric,
                cluster_selection_method=self.cfg.cluster_selection_method,
                allow_single_cluster=self.cfg.allow_single_cluster,
                core_dist_n_jobs=-1,
                prediction_data=False,
            )
            self.model.fit(X)
            labels = self.model.labels_
            probs = getattr(self.model, "probabilities_", np.ones_like(labels, dtype=float))
            outlier = getattr(self.model, "outlier_scores_", np.zeros_like(labels, dtype=float))
            algo = "hdbscan"
        else:
            # Fallback to DBSCAN
            self.model = DBSCAN(eps=0.5, min_samples=self.cfg.min_cluster_size, n_jobs=-1)
            self.model.fit(X)
            labels = self.model.labels_
            probs = np.ones_like(labels, dtype=float)
            outlier = np.zeros_like(labels, dtype=float)
            algo = "dbscan_fallback"

        self.meta = {
            "algo": algo,
            "config": asdict(self.cfg),
            "features": self.feature_names,
            "n_samples": int(len(labels)),
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        }
        return {
            "labels": labels,
            "probabilities": probs,
            "outlier_scores": outlier,
            "meta": self.meta,
        }

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, out / "scaler.joblib")
        if self.model is not None:
            joblib.dump(self.model, out / "model.joblib")
        meta_path = out / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            import json

            json.dump(self.meta, f, indent=2)

