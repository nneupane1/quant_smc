"""
Legacy compatibility surface for liquidity HDBSCAN training.

This keeps the CLI-facing `quant_system.models.liquidity.hdbscan_trainer`
API intact while delegating to the canonical `ml.regime` implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import pandas as pd
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
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = True
    features: Optional[List[str]] = None
    seed: int = 42
    emit_labels: bool = True


class HDBSCANTrainer:
    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self._trainer = _CanonicalTrainer(
            _CanonicalConfig(
                timeframe="15m",
                feature_cols=cfg.features,
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_method=cfg.cluster_selection_method,
                metric=cfg.metric,
                allow_single_cluster=cfg.allow_single_cluster,
                seed=cfg.seed,
            )
        )
        self.features_: List[str] = []
        self.stats_: Dict[str, Any] = {}
        self.labels_: Optional[pd.DataFrame] = None
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        states = self._trainer.fit(df)
        self.features_ = list(self._trainer.columns_)
        self.labels_ = states.rename(columns={"is_outlier": "outlier_score"})
        meta = self._trainer.meta_
        self.stats_ = {
            "n_clusters": int(meta.get("n_clusters", 0)),
            "n_points": int(len(states)),
            "noise": int(round(meta.get("noise_rate", 0.0) * len(states))),
            "labels_unique": sorted(states["cluster_id"].astype(int).unique().tolist()),
        }
        return self.stats_

    def save(self, out_dir: str):
        if self.labels_ is None:
            raise RuntimeError("Model not fit.")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._trainer.save(str(out), self.labels_.rename(columns={"outlier_score": "is_outlier"}))
        self.model = self._trainer.model
        self.scaler = self._trainer.scaler
        meta = {
            "cfg": self.cfg.__dict__,
            "features": self.features_,
            "stats": self.stats_,
        }
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        if self.cfg.emit_labels and self.labels_ is not None:
            self.labels_.to_csv(out / "labels.csv", index=False)

    @staticmethod
    def load(model_dir: str):
        meta_path = Path(model_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {model_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = HDBSCANTrainer(HDBSCANConfig(**meta.get("cfg", {})))
        obj.features_ = meta.get("features", [])
        obj.stats_ = meta.get("stats", {})
        model_path = Path(model_dir) / "model.joblib"
        scaler_path = Path(model_dir) / "scaler.joblib"
        if model_path.exists():
            obj.model = joblib.load(model_path)
        if scaler_path.exists():
            obj.scaler = joblib.load(scaler_path)
        labels_path = Path(model_dir) / "labels.csv"
        if labels_path.exists():
            obj.labels_ = pd.read_csv(labels_path)
        return obj

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.features_:
            raise RuntimeError("Model not fit.")
        if self.model is None or self.scaler is None:
            raise RuntimeError("Loaded wrapper does not have model/scaler artifacts available.")
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
