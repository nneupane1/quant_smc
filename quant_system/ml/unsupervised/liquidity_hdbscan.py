"""
Compatibility wrapper for the canonical 15m/1h HDBSCAN liquidity trainer.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd

from quant_system.ml.regime.hdbscan_trainer import (
    HDBSCANConfig as _CanonicalConfig,
    LiquidityClusterTrainer as _CanonicalTrainer,
)


@dataclass
class LiquidityHDBSCANConfig:
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "leaf"
    allow_single_cluster: bool = True
    seed: int = 42
    timeframe: str = "15m"
    feature_cols: Optional[List[str]] = None


class LiquidityClusterer:
    def __init__(self, cfg: LiquidityHDBSCANConfig):
        self.cfg = cfg
        self._trainer = _CanonicalTrainer(
            _CanonicalConfig(
                timeframe=cfg.timeframe,
                feature_cols=cfg.feature_cols,
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_method=cfg.cluster_selection_method,
                metric=cfg.metric,
                allow_single_cluster=cfg.allow_single_cluster,
                seed=cfg.seed,
            )
        )
        self.feature_names: List[str] = []
        self.meta: Dict = {}

    def fit(self, df: pd.DataFrame) -> Dict:
        clusters = self._trainer.fit(df)
        self.feature_names = list(self._trainer.columns_)
        labels = clusters["cluster_id"].astype(int).to_numpy()
        outlier = clusters["is_outlier"].astype(float).to_numpy()
        self.meta = {
            "algo": "hdbscan",
            "config": asdict(self.cfg),
            "features": self.feature_names,
            "n_samples": int(len(labels)),
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        }
        return {
            "labels": labels,
            "probabilities": np.ones_like(labels, dtype=float),
            "outlier_scores": outlier,
            "meta": self.meta,
        }

    def save(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
