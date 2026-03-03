"""
Unsupervised regime detector using a Gaussian HMM on higher-TF bars (6h/12h).

Features (per bar):
 - log return (close-to-close)
 - range_pct  = (high-low)/close
 - realized_vol (rolling std of log returns)
 - dollar_vol = close * volume

Outputs:
 - state: argmax posterior state id per bar
 - state_post_{k}: posterior probabilities for each state

Usage:
    from quant_system.ml.training.regime_hmm_trainer import RegimeHMMTrainer, RegimeHMMConfig
    df = pd.read_csv("XBTUSD_6h.csv", parse_dates=["dt"])
    trainer = RegimeHMMTrainer(RegimeHMMConfig(n_states=5))
    res = trainer.fit_transform(df)
    trainer.save("models/regime_hmm_XBTUSD_6h")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeHMMConfig:
    """Configuration for the HMM regime trainer."""

    n_states: int = 5
    covariance_type: str = "diag"
    seed: int = 42
    vol_window: int = 64  # bars for realized vol
    feature_cols: Optional[List[str]] = None  # custom feature names (must exist in df)


class RegimeHMMTrainer:
    """
    Fits an HMM on 6h/12h bars and emits per-bar regime posteriors.
    """

    def __init__(self, cfg: Optional[RegimeHMMConfig] = None):
        self.cfg = cfg or RegimeHMMConfig()
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[GaussianHMM] = None
        self.features_: List[str] = []

    def _init_hmm_params(self, X: np.ndarray):
        n_states = int(self.cfg.n_states)
        n_features = X.shape[1]
        idx = np.linspace(0, len(X) - 1, n_states, dtype=int)
        means = X[idx].copy()

        cov = np.cov(X, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = np.asarray(cov, dtype=float)
        cov = np.nan_to_num(cov, nan=1e-6, posinf=1e-6, neginf=1e-6)
        cov = cov + np.eye(n_features) * 1e-6
        diag = np.clip(np.diag(cov), 1e-6, None)

        startprob = np.full(n_states, 1.0 / n_states, dtype=float)
        transmat = np.full((n_states, n_states), 1.0 / n_states, dtype=float)

        if self.cfg.covariance_type == "full":
            covars = np.repeat(cov[None, :, :], n_states, axis=0)
        elif self.cfg.covariance_type == "diag":
            covars = np.repeat(diag[None, :], n_states, axis=0)
        elif self.cfg.covariance_type == "spherical":
            covars = np.repeat(np.array([float(diag.mean())], dtype=float), n_states)
        elif self.cfg.covariance_type == "tied":
            covars = cov
        else:
            covars = np.repeat(diag[None, :], n_states, axis=0)

        return startprob, transmat, means, covars

    # ------------------------------------------------------------------ #
    def _build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build the feature matrix X (numpy-ready) and aligned dataframe with dt.
        Expects columns: dt, close, high, low, volume (lowercase).
        """
        frame = df.copy()
        frame = frame.sort_values("dt").reset_index(drop=True)

        if self.cfg.feature_cols:
            # user-specified feature list; ensure numeric
            missing = [c for c in self.cfg.feature_cols if c not in frame.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            feat_df = frame[self.cfg.feature_cols].apply(pd.to_numeric, errors="coerce")
        else:
            for col in ("close", "high", "low", "volume"):
                if col not in frame.columns:
                    raise ValueError(f"Required column '{col}' not found.")
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

            frame["logret"] = np.log(frame["close"]).diff()
            frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
            frame["realized_vol"] = frame["logret"].rolling(self.cfg.vol_window, min_periods=self.cfg.vol_window // 2).std()
            frame["dollar_vol"] = frame["close"] * frame["volume"]

            feat_df = frame[["logret", "range_pct", "realized_vol", "dollar_vol"]]

        feat_df = feat_df.dropna()
        dt_aligned = frame.loc[feat_df.index, ["dt"]].reset_index(drop=True)
        feat_df = feat_df.reset_index(drop=True)
        return feat_df, dt_aligned

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "RegimeHMMTrainer":
        """
        Fit the HMM on provided dataframe of higher-TF bars.
        """
        feat_df, _ = self._build_features(df)
        self.features_ = feat_df.columns.tolist()

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feat_df.values)

        self.model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            random_state=self.cfg.seed,
            n_iter=200,
            init_params="",
            verbose=False,
        )
        startprob, transmat, means, covars = self._init_hmm_params(X)
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars
        self.model.fit(X)
        return self

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute posterior regime probabilities and state ids for the given dataframe.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        feat_df, dt_aligned = self._build_features(df)
        X = self.scaler.transform(feat_df.values)

        post = self.model.predict_proba(X)
        states = post.argmax(axis=1)

        out = dt_aligned.copy()
        out["state"] = states
        for k in range(post.shape[1]):
            out[f"state_post_{k}"] = post[:, k]
        return out

    # ------------------------------------------------------------------ #
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------ #
    def save(self, out_dir: str):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted; cannot save.")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        meta = {
            "config": asdict(self.cfg),
            "features": self.features_,
        }
        with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        joblib.dump(self.model, Path(out_dir) / "model.joblib")
        joblib.dump(self.scaler, Path(out_dir) / "scaler.joblib")

    @staticmethod
    def load(model_dir: str) -> "RegimeHMMTrainer":
        meta_path = Path(model_dir) / "meta.json"
        model_path = Path(model_dir) / "model.joblib"
        scaler_path = Path(model_dir) / "scaler.joblib"

        if not meta_path.exists() or not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Missing model artifacts in directory.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = RegimeHMMConfig(**meta["config"])
        obj = RegimeHMMTrainer(cfg)
        obj.features_ = meta.get("features", [])
        obj.model = joblib.load(model_path)
        obj.scaler = joblib.load(scaler_path)
        return obj
