"""
HDBSCAN-based microstructure/liquidity clustering (15m / 1h).

Produces cluster labels and outlier flags per bar for downstream gating.
Saves:
  - model.joblib (HDBSCAN instance)
  - scaler.joblib (StandardScaler)
  - meta.json (config, columns, params, n_clusters, noise rate)
  - states.csv (dt, cluster_id, is_outlier)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


@dataclass
class HDBSCANConfig:
    timeframe: str = "15m"            # "15m" or "1h"
    feature_cols: Optional[List[str]] = None
    min_cluster_size: int = 30
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    metric: str = "euclidean"
    allow_single_cluster: bool = False
    seed: int = 42


class LiquidityClusterTrainer:
    def __init__(self, cfg: HDBSCANConfig):
        if hdbscan is None:
            raise ImportError("hdbscan is required for liquidity clustering")
        self.cfg = cfg
        self.model: Optional[hdbscan.HDBSCAN] = None
        self.scaler: Optional[StandardScaler] = None
        self.columns_: List[str] = []
        self.meta_: Dict[str, Any] = {}

    # --------------- data prep --------------- #
    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        tf = self.cfg.timeframe.lower()
        if tf not in {"15m", "1h"}:
            raise ValueError("timeframe must be 15m or 1h")
        if "dt" not in df.columns:
            raise ValueError("Input must have dt column")
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df = df.set_index("dt")
        rule = "15T" if tf == "15m" else "1H"
        # numeric columns mean; for non-numeric, drop
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        res = df[num_cols].resample(rule, label="right", closed="right").mean()
        res = res.dropna()
        res = res.reset_index()
        return res

    def _select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.feature_cols:
            cols = [c for c in self.cfg.feature_cols if c in df.columns]
        else:
            # default micro/liquidity/vol set
            candidate = [
                "volume",
                "dollar_volume",
                "atr",
                "vol_pctile",
                "range_pct",
                "liq_sweep_intensity",
                "spread_bps",
                "absorption_score",
                "toxicity",
                "vol_zscore",
            ]
            cols = [c for c in candidate if c in df.columns]
        if not cols:
            raise ValueError("No usable feature columns found for clustering.")
        return df[cols], cols

    # --------------- fit --------------- #
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._resample(df)
        X, cols = self._select_cols(df)
        self.columns_ = cols

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            cluster_selection_method=self.cfg.cluster_selection_method,
            metric=self.cfg.metric,
            allow_single_cluster=self.cfg.allow_single_cluster,
            core_dist_n_jobs= -1,
        )
        labels = self.model.fit_predict(Xs)
        df_out = pd.DataFrame(
            {
                "dt": df["dt"].values,
                "cluster_id": labels.astype(int),
                "is_outlier": (labels == -1).astype(int),
            }
        )

        n_clusters = len(set([l for l in labels if l >= 0]))
        noise_rate = float((labels == -1).mean())
        self.meta_ = {
            "config": asdict(self.cfg),
            "features": cols,
            "n_clusters": n_clusters,
            "noise_rate": noise_rate,
        }
        return df_out

    # --------------- persistence --------------- #
    def save(self, out_dir: str, states: pd.DataFrame):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fit.")
        joblib.dump(self.model, Path(out_dir) / "model.joblib")
        joblib.dump(self.scaler, Path(out_dir) / "scaler.joblib")
        with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta_, f, indent=2)
        states.to_csv(Path(out_dir) / "states.csv", index=False)
