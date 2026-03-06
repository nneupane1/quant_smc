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
from typing import List, Optional, Dict, Any, Tuple

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

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
    scaler: str = "robust"  # robust | standard
    clip_quantiles: Tuple[float, float] = (0.005, 0.995)


class LiquidityClusterTrainer:
    def __init__(self, cfg: HDBSCANConfig):
        if hdbscan is None:
            raise ImportError("hdbscan is required for liquidity clustering")
        self.cfg = cfg
        self.model: Optional[hdbscan.HDBSCAN] = None
        self.scaler: Optional[object] = None
        self.columns_: List[str] = []
        self.meta_: Dict[str, Any] = {}
        self.fill_values_: Optional[pd.Series] = None
        self.clip_lower_: Optional[pd.Series] = None
        self.clip_upper_: Optional[pd.Series] = None

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
        rule = "15min" if tf == "15m" else "1h"
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

    def _build_scaler(self):
        mode = str(getattr(self.cfg, "scaler", "robust")).strip().lower()
        if mode == "standard":
            return StandardScaler()
        return RobustScaler()

    def _prepare_matrix(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        matrix = X.copy().apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)

        lower_q, upper_q = self.cfg.clip_quantiles
        lower_q = float(min(max(lower_q, 0.0), 0.49))
        upper_q = float(max(min(upper_q, 1.0), 0.51))

        if fit:
            self.fill_values_ = matrix.median(axis=0, numeric_only=True).fillna(0.0)
            self.clip_lower_ = matrix.quantile(lower_q, interpolation="linear").fillna(self.fill_values_)
            self.clip_upper_ = matrix.quantile(upper_q, interpolation="linear").fillna(self.fill_values_)

        if self.fill_values_ is None:
            self.fill_values_ = matrix.median(axis=0, numeric_only=True).fillna(0.0)
        if self.clip_lower_ is None or self.clip_upper_ is None:
            self.clip_lower_ = matrix.quantile(lower_q, interpolation="linear").fillna(self.fill_values_)
            self.clip_upper_ = matrix.quantile(upper_q, interpolation="linear").fillna(self.fill_values_)

        matrix = matrix.fillna(self.fill_values_)
        matrix = matrix.clip(lower=self.clip_lower_, upper=self.clip_upper_, axis=1)
        matrix = matrix.fillna(0.0)
        return matrix

    # --------------- fit --------------- #
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._resample(df)
        X, cols = self._select_cols(df)
        X = self._prepare_matrix(X, fit=True)
        self.columns_ = cols

        self.scaler = self._build_scaler()
        Xs = self.scaler.fit_transform(X.values)

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            cluster_selection_method=self.cfg.cluster_selection_method,
            metric=self.cfg.metric,
            allow_single_cluster=self.cfg.allow_single_cluster,
            prediction_data=True,
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
            "preprocessing": {
                "fill_values": self.fill_values_.to_dict() if self.fill_values_ is not None else {},
                "clip_lower": self.clip_lower_.to_dict() if self.clip_lower_ is not None else {},
                "clip_upper": self.clip_upper_.to_dict() if self.clip_upper_ is not None else {},
            },
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
