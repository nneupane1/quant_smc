"""
Unsupervised regime detection using Gaussian HMM (6h/12h).

Outputs state_id per bar and persists model/scaler/metadata.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict
import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from joblib import dump

from quant_system.utils.logger import get_logger

LOG = get_logger("regime_hmm")


@dataclass
class HMMConfig:
    n_states: int = 4
    n_iter: int = 200
    covariance_type: str = "full"
    seed: int = 42
    feature_cols: Optional[List[str]] = None  # defaults set in trainer if None
    tf: str = "6h"  # tag for the timeframe


class RegimeHMMTrainer:
    """
    Fits a Gaussian HMM on selected features and returns state_id per row.
    """

    DEFAULT_COLS = [
        "close",
        "atr",
        "realized_vol",
        "range_pct",
        "volatility_zscore",
        "volume",
    ]

    def __init__(self, cfg: Optional[HMMConfig] = None):
        self.cfg = cfg or HMMConfig()
        if self.cfg.feature_cols is None:
            self.cfg.feature_cols = list(self.DEFAULT_COLS)
        self.scaler = StandardScaler()
        self.model: Optional[GaussianHMM] = None
        self.feature_cols_used: List[str] = []

    # ----------------------------------------------------- #
    def _prepare_features(self, df: pd.DataFrame) -> (np.ndarray, pd.Index):
        cols = [c for c in self.cfg.feature_cols if c in df.columns]
        if not cols:
            raise ValueError("No overlap between requested feature_cols and dataframe columns.")
        self.feature_cols_used = cols
        X = df[cols].astype(float)
        # drop rows with NaN to keep HMM happy; track index
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask]
        return X.values, X.index

    # ----------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> "RegimeHMMTrainer":
        X, _ = self._prepare_features(df)
        Xs = self.scaler.fit_transform(X)
        self.model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.seed,
        )
        self.model.fit(Xs)
        LOG.info("[RegimeHMM] Fitted HMM | states=%s iters=%s", self.cfg.n_states, self.cfg.n_iter)
        return self

    # ----------------------------------------------------- #
    def predict_states(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X, idx = self._prepare_features(df)
        Xs = self.scaler.transform(X)
        states = self.model.predict(Xs)
        out = pd.DataFrame({"state_id": states, "tf": self.cfg.tf}, index=idx)
        if "dt" in df.columns:
            out = out.join(df["dt"])
        return out.reset_index(drop=True)

    # ----------------------------------------------------- #
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.predict_states(df)

    # ----------------------------------------------------- #
    def save(self, out_dir: str):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dump(self.model, out_path / "model.joblib")
        dump(self.scaler, out_path / "scaler.joblib")
        meta: Dict = {
            "config": asdict(self.cfg),
            "feature_cols_used": self.feature_cols_used,
        }
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        LOG.info("[RegimeHMM] Saved artifacts to %s", out_path)

