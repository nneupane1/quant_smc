"""
Compatibility wrapper for the canonical 15m/1h HDBSCAN liquidity trainer.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd

from quant_system.ml.regime.hdbscan_trainer import (
    HDBSCANConfig as _CanonicalConfig,
    LiquidityClusterTrainer,
)


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 15
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    features: Optional[List[str]] = None
    timeframe: str = "15m"


class HDBSCANClusterer:
    def __init__(self, cfg: HDBSCANConfig):
        self.cfg = cfg
        self._trainer = LiquidityClusterTrainer(
            _CanonicalConfig(
                timeframe=cfg.timeframe,
                feature_cols=cfg.features,
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_epsilon=cfg.cluster_selection_epsilon,
                cluster_selection_method=cfg.cluster_selection_method,
                metric=cfg.metric,
                allow_single_cluster=cfg.allow_single_cluster,
            )
        )
        self.selected_features_: List[str] = []
        self.report_: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        clusters = self._trainer.fit(df)
        self.selected_features_ = list(self._trainer.columns_)
        vals, counts = np.unique(clusters["cluster_id"], return_counts=True)
        self.report_ = {
            "params": asdict(self.cfg),
            "features": self.selected_features_,
            "n_obs": int(len(clusters)),
            "clusters": {int(v): int(c) for v, c in zip(vals, counts)},
        }
        out = clusters.rename(columns={"cluster_id": "cluster"})
        return out

    def save(self, out_dir: Path, asset: str, tf: str, clusters: pd.DataFrame):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{asset}_{tf}_clusters.csv"
        clusters.to_csv(out_csv, index=False)
        with open(out_dir / "train_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report_, f, indent=2)
