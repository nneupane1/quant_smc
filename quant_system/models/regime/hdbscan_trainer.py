"""
Legacy compatibility surface for HDBSCAN liquidity/regime clustering.

Delegates to the canonical implementation in
`quant_system.ml.regime.hdbscan_trainer`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import json
import joblib

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None

from quant_system.ml.regime.hdbscan_trainer import (
    HDBSCANConfig as _CanonicalConfig,
    LiquidityClusterTrainer as _CanonicalTrainer,
)


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
    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self._trainer = _CanonicalTrainer(
            _CanonicalConfig(
                timeframe="15m",
                feature_cols=cfg.features,
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_epsilon=cfg.cluster_selection_epsilon,
                cluster_selection_method=cfg.cluster_selection_method,
                metric=cfg.metric,
                allow_single_cluster=cfg.allow_single_cluster,
                seed=cfg.random_state or 42,
            )
        )
        self.labels_: Optional[pd.DataFrame] = None
        self.features_: List[str] = []
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        states = self._trainer.fit(df)
        self.labels_ = states.rename(
            columns={"cluster_id": "cluster_id", "is_outlier": "is_outlier"}
        )
        self.features_ = list(self._trainer.columns_)
        self.model = self._trainer.model
        self.scaler = self._trainer.scaler
        meta = self._trainer.meta_
        return {
            "n_clusters": int(meta.get("n_clusters", 0)),
            "n_noise": int(round(meta.get("noise_rate", 0.0) * len(states))),
            "features": list(meta.get("features", [])),
        }

    def save(self, out_dir: str):
        if self.labels_ is None:
            raise RuntimeError("Call fit() before save().")
        self._trainer.save(out_dir, self.labels_)
        out = Path(out_dir)
        meta = {
            "cfg": self.cfg.__dict__,
            "features": self.features_,
        }
        with open(out / "legacy_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(model_dir: str) -> "HDBSCANTrainer":
        meta_path = Path(model_dir) / "legacy_meta.json"
        cfg = HDBSCANConfig()
        features: List[str] = []
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = HDBSCANConfig(**meta.get("cfg", {}))
            features = meta.get("features", [])
        obj = HDBSCANTrainer(cfg)
        obj.features_ = features
        model_path = Path(model_dir) / "model.joblib"
        scaler_path = Path(model_dir) / "scaler.joblib"
        if model_path.exists():
            obj.model = joblib.load(model_path)
        if scaler_path.exists():
            obj.scaler = joblib.load(scaler_path)
        labels_path = Path(model_dir) / "states.csv"
        if labels_path.exists():
            obj.labels_ = pd.read_csv(labels_path)
        return obj

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None or not self.features_:
            raise RuntimeError("Model not loaded or fitted.")
        if hdbscan is None:
            raise ImportError("hdbscan is required for approximate prediction.")
        feats = [c for c in self.features_ if c in df.columns]
        if not feats:
            raise ValueError("None of the trained feature columns are present in the dataframe.")
        X = df[feats].astype(float).fillna(0.0)
        Xs = self.scaler.transform(X)
        labels, strengths = hdbscan.approximate_predict(self.model, Xs)
        out = pd.DataFrame(index=df.index)
        out["cluster_id"] = labels.astype(int)
        out["prob"] = strengths.astype(float)
        return out
