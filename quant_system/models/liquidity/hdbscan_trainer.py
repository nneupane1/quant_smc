"""
Unsupervised liquidity/chop clustering using HDBSCAN.

Typical usage:
    python -m quant_system.cli.train_hdbscan --input data/features_xbtusd/XBTUSD_features.csv \
        --out-dir models/liquidity_hdbscan_xbtusd_15m

Outputs:
 - model.joblib          : fitted HDBSCAN model
 - meta.json             : config, features used, basic fit stats
 - labels.csv (optional) : dt, cluster_id, prob, outlier_score (for inspection)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


try:
    import hdbscan
except ImportError as e:  # pragma: no cover - handled by user install
    raise ImportError("hdbscan is required. Please install via `pip install hdbscan`.") from e


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = True
    features: Optional[List[str]] = None  # if None, a sensible default list is used
    seed: int = 42
    emit_labels: bool = True


class HDBSCANTrainer:
    """
    Fits HDBSCAN on liquidity/vol/dispersion features.
    """

    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self.model: Optional[hdbscan.HDBSCAN] = None
        self.features_: List[str] = []
        self.stats_: Dict[str, Any] = {}

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.features:
            cols = [c for c in self.cfg.features if c in df.columns]
        else:
            # Default liquidity/vol/dispersion set (only keep if present)
            candidates = [
                "volume",
                "dollar_volume",
                "atr",
                "range_pct",
                "realized_vol",
                "vol_zscore",
                "absorption_score",
                "spread",  # if present
                "tick_density",  # if present
                "liquidity_density",  # if present
            ]
            cols = [c for c in candidates if c in df.columns]
        if not cols:
            raise ValueError("No valid feature columns found for HDBSCAN clustering.")
        self.features_ = cols
        return df[cols].astype(float)

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = df.copy()
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
        X = self._select_features(df)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric=self.cfg.metric,
            cluster_selection_method=self.cfg.cluster_selection_method,
            allow_single_cluster=self.cfg.allow_single_cluster,
            prediction_data=True,
            gen_min_span_tree=False,
        )
        labels = model.fit_predict(Xs)
        probs = model.probabilities_
        outlier = model.outlier_scores_

        self.model = model
        self.scaler = scaler

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.stats_ = {
            "n_clusters": int(n_clusters),
            "n_points": int(len(labels)),
            "noise": int((labels == -1).sum()),
            "labels_unique": sorted(list(map(int, np.unique(labels)))),
        }

        result = pd.DataFrame(
            {
                "dt": df["dt"] if "dt" in df.columns else np.arange(len(labels)),
                "cluster_id": labels.astype(int),
                "prob": probs.astype(float),
                "outlier_score": outlier.astype(float),
            }
        )
        self.labels_ = result
        return self.stats_

    def save(self, out_dir: str):
        if self.model is None:
            raise RuntimeError("Model not fit.")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # save model and scaler together
        joblib.dump({"model": self.model, "scaler": self.scaler}, out / "model.joblib")
        meta = {
            "cfg": asdict(self.cfg),
            "features": self.features_,
            "stats": self.stats_,
        }
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        if self.cfg.emit_labels and hasattr(self, "labels_"):
            self.labels_.to_csv(out / "labels.csv", index=False)

    @staticmethod
    def load(model_dir: str) -> "HDBSCANTrainer":
        with open(Path(model_dir) / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = HDBSCANConfig(**meta["cfg"])
        obj = HDBSCANTrainer(cfg)
        bundle = joblib.load(Path(model_dir) / "model.joblib")
        obj.model = bundle["model"]
        obj.scaler = bundle["scaler"]
        obj.features_ = meta.get("features", [])
        obj.stats_ = meta.get("stats", {})
        return obj

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign clusters to new data (approximate prediction).
        """
        if self.model is None:
            raise RuntimeError("Model not fit.")
        X = self._select_features(df)
        Xs = self.scaler.transform(X)
        labels, strengths = hdbscan.approximate_predict(self.model, Xs)
        return pd.DataFrame(
            {
                "cluster_id": labels.astype(int),
                "prob": strengths.astype(float),
            },
            index=df.index,
        )
