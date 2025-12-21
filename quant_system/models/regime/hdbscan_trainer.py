"""
HDBSCAN-based liquidity/microstructure clustering.

This trains an unsupervised HDBSCAN model on 15m/1h liquidity & volatility
features to discover chop/acceleration/anomaly states. It saves:
  - model.joblib      (HDBSCAN)
  - scaler.joblib     (StandardScaler on selected features)
  - meta.json         (config, features, summary stats)
  - labels.csv        (dt, cluster, prob/strength)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover - dependency guard
    raise ImportError(
        "hdbscan is required. Install with `pip install hdbscan`."
    ) from e


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    features: Optional[List[str]] = None
    random_state: Optional[int] = 42


class HDBSCANTrainer:
    """
    Fits an HDBSCAN clusterer on liquidity/volatility features.
    """

    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self.model_: Optional[hdbscan.HDBSCAN] = None
        self.scaler_: Optional[StandardScaler] = None
        self.features_: List[str] = []
        self.summary_: Dict = {}

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Default feature set if none provided
        default_features = [
            "range_pct",
            "atr",
            "volatility_zscore",
            "dollar_volume",
            "volume",
            "absorption_score",
            "wick_pressure",
            "fvg_open_total",
            "fvg_ctx_weight",
        ]
        feats = self.cfg.features or default_features
        present = [c for c in feats if c in df.columns]
        if not present:
            raise ValueError("No requested HDBSCAN features found in dataframe.")
        self.features_ = present
        return df[present].copy()

    def fit(self, df: pd.DataFrame) -> Dict:
        """
        Fit HDBSCAN on the provided dataframe.
        df must contain a 'dt' column (datetime) and the feature columns.
        """
        if "dt" not in df.columns:
            raise ValueError("Input dataframe must contain 'dt' column.")

        X = self._select_features(df)
        X = X.replace([np.inf, -np.inf], np.nan)
        # Drop rows with missing feature values
        mask = X.notna().all(axis=1)
        X = X[mask]
        dt_used = df.loc[mask, "dt"].reset_index(drop=True)

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)

        self.model_ = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric=self.cfg.metric,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            cluster_selection_method=self.cfg.cluster_selection_method,
            allow_single_cluster=self.cfg.allow_single_cluster,
            core_dist_n_jobs=-1,
        )
        labels = self.model_.fit_predict(Xs)
        probs = getattr(self.model_, "probabilities_", np.ones_like(labels, dtype=float))
        strengths = getattr(self.model_, "outlier_scores_", np.zeros_like(labels, dtype=float))

        # Build summary
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        n_noise = int(cluster_counts.get(-1, 0))
        n_clusters = len([k for k in unique if k != -1])

        self.summary_ = {
            "n_samples": int(len(labels)),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_counts": cluster_counts,
            "features_used": self.features_,
        }

        # store labels dataframe
        self.labels_df_ = pd.DataFrame(
            {
                "dt": pd.to_datetime(dt_used).astype(str),
                "cluster": labels,
                "probability": probs,
                "outlier_strength": strengths,
            }
        )
        return self.summary_

    def save(self, out_dir: str):
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Call fit() before save().")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_, out_path / "model.joblib")
        joblib.dump(self.scaler_, out_path / "scaler.joblib")
        meta = {
            "config": asdict(self.cfg),
            "features": self.features_,
            "summary": self.summary_,
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if hasattr(self, "labels_df_"):
            self.labels_df_.to_csv(out_path / "labels.csv", index=False)

    @staticmethod
    def load(model_dir: str) -> "HDBSCANTrainer":
        model_dir_path = Path(model_dir)
        with open(model_dir_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = HDBSCANConfig(**meta["config"])
        obj = HDBSCANTrainer(cfg)
        obj.model_ = joblib.load(model_dir_path / "model.joblib")
        obj.scaler_ = joblib.load(model_dir_path / "scaler.joblib")
        obj.features_ = meta.get("features", [])
        obj.summary_ = meta.get("summary", {})
        labels_path = model_dir_path / "labels.csv"
        if labels_path.exists():
            obj.labels_df_ = pd.read_csv(labels_path)
        return obj

    # Inference helpers
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model not loaded or fitted.")
        X = df[self.features_].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        Xs = self.scaler_.transform(X)
        return self.model_.approximate_predict(Xs)[0]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model not loaded or fitted.")
        X = df[self.features_].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        Xs = self.scaler_.transform(X)
        return self.model_.approximate_predict(Xs)[1]
