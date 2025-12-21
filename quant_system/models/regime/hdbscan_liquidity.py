"""
Unsupervised liquidity / microstructure clustering via HDBSCAN.

Clustering is done on 15m or 1h bars using liquidity/volatility/dispersion
features. Outputs:
  - cluster_id (integer label; -1 for outlier)
  - is_outlier (1 if outlier)
  - cluster_prob (membership strength from HDBSCAN)

Saved artifacts (in model_dir):
  - model.joblib
  - meta.json (config, feature list)
  - Optional: states.csv (dt, cluster_id, is_outlier, cluster_prob) if requested by CLI.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import hdbscan


# --------------------------- Config --------------------------- #
@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "leaf"
    allow_single_cluster: bool = False
    features: Optional[List[str]] = None  # if None, we auto-pick


# --------------------------- Trainer --------------------------- #
class HDBSCANLiquidity:
    """
    Fit HDBSCAN on liquidity/volatility/dispersion features.
    """

    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self.features_: List[str] = []
        self.model_: Optional[hdbscan.HDBSCAN] = None

    def _auto_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "volume",
            "dollar_volume",
            "atr",
            "range_pct",
            "volatility",
            "realized_vol",
            "spread",
            "imbalance",
            "liquidity_pressure",
        ]
        return [c for c in candidates if c in df.columns]

    def fit(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        if "dt" not in df.columns:
            raise ValueError("Expected 'dt' column in dataframe.")

        feats = self.cfg.features or self._auto_features(df)
        if not feats:
            raise ValueError("No usable features found for HDBSCAN clustering.")
        self.features_ = feats

        X = df[feats].astype(float).fillna(0.0).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric=self.cfg.metric,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            cluster_selection_method=self.cfg.cluster_selection_method,
            allow_single_cluster=self.cfg.allow_single_cluster,
        )
        labels = model.fit_predict(Xs)
        probs = model.probabilities_

        self.model_ = Pipeline([("scaler", scaler), ("hdb", model)])

        outlier_frac = float(np.mean(labels == -1)) if len(labels) else 0.0
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        return {
            "n_clusters": n_clusters,
            "outlier_frac": outlier_frac,
            "features": feats,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        X = df[self.features_].astype(float).fillna(0.0).values
        labels = self.model_.named_steps["hdb"].approximate_predict(
            self.model_.named_steps["hdb"], self.model_.named_steps["scaler"].transform(X)
        )  # type: ignore
        # approximate_predict returns (labels, probs)
        lab, prob = labels
        out = pd.DataFrame(
            {
                "cluster_id": lab,
                "cluster_prob": prob,
                "is_outlier": (lab == -1).astype(int),
            },
            index=df.index,
        )
        return out

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df merged with cluster outputs."""
        preds = self.predict(df)
        return df.join(preds)

    def save(self, model_dir: str):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        meta = {
            "cfg": asdict(self.cfg),
            "features": self.features_,
        }
        with open(Path(model_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        joblib.dump(self.model_, Path(model_dir) / "model.joblib")

    @staticmethod
    def load(model_dir: str) -> "HDBSCANLiquidity":
        with open(Path(model_dir) / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = HDBSCANConfig(**meta["cfg"])
        obj = HDBSCANLiquidity(cfg)
        obj.features_ = meta.get("features", [])
        obj.model_ = joblib.load(Path(model_dir) / "model.joblib")
        return obj

