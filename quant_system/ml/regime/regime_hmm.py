"""
Regime HMM trainer (unsupervised) for 6h/12h timeframes.

- Builds leak-safe feature set (log returns, range pct, realized vol, volume z).
- Fits Gaussian HMM (hmmlearn) on scaled features.
- Saves model, scaler, meta, and decoded state timeline.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeHMMConfig:
    n_states: int = 5
    covariance: str = "diag"
    vol_window: int = 24       # realized vol window (bars)
    max_iter: int = 300
    random_state: int = 42
    tf_label: str = "6h"
    asset: Optional[str] = None


class RegimeHMMTrainer:
    def __init__(self, cfg: RegimeHMMConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[hmm.GaussianHMM] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}

    # ------------------------ feature builder ------------------------ #
    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        required = ["dt", "close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for regime HMM.")

        frame = df.sort_values("dt").reset_index(drop=True).copy()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
        frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

        log_ret = np.log(frame["close"]).diff()
        range_pct = (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)
        realized_vol = log_ret.rolling(self.cfg.vol_window).std()
        vol_z = (frame["volume"] - frame["volume"].mean()) / frame["volume"].std()

        feat_df = pd.DataFrame(
            {
                "log_ret": log_ret,
                "range_pct": range_pct,
                "realized_vol": realized_vol,
                "vol_z": vol_z,
            }
        )
        feat_df = feat_df.dropna()
        self.feature_names = feat_df.columns.tolist()
        X = feat_df.values.astype(float)
        dts = frame.loc[feat_df.index, "dt"].tolist()
        return X, dts

    # ------------------------ fit ------------------------ #
    def fit(self, df: pd.DataFrame) -> Dict:
        X_raw, dts = self._build_features(df)
        if len(X_raw) < self.cfg.n_states * 5:
            raise ValueError("Not enough samples to fit HMM; reduce n_states or provide more data.")

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X_raw)

        self.model = hmm.GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance,
            n_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            verbose=False,
        )
        self.model.fit(X)

        loglik = float(self.model.score(X))
        n = len(X)
        k = self.cfg.n_states
        n_features = X.shape[1]
        n_params = k * (k - 1) + k * (n_features * 2)  # transitions + means/vars
        aic = 2 * n_params - 2 * loglik
        bic = n_params * np.log(n) - 2 * loglik

        self.metrics = {
            "log_likelihood": loglik,
            "aic": float(aic),
            "bic": float(bic),
            "n": int(n),
            "n_states": int(self.cfg.n_states),
            "tf": self.cfg.tf_label,
        }

        states = self.model.predict(X)
        post = self.model.predict_proba(X)
        post_df = pd.DataFrame(post, columns=[f"p_state{i}" for i in range(self.cfg.n_states)])
        self.decoded_df = pd.DataFrame({"dt": dts, "state": states}).join(post_df)
        return {"metrics": self.metrics}

    # ------------------------ inference ------------------------ #
    def predict_states(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted.")
        X_raw, dts = self._build_features(df)
        X = self.scaler.transform(X_raw)
        states = self.model.predict(X)
        post = self.model.predict_proba(X)
        post_df = pd.DataFrame(post, columns=[f"p_state{i}" for i in range(self.cfg.n_states)])
        return pd.DataFrame({"dt": dts, "state": states}).join(post_df)

    # ------------------------ persistence ------------------------ #
    def save(self, out_dir: str):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() before save().")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, Path(out_dir) / "model.joblib")
        joblib.dump(self.scaler, Path(out_dir) / "scaler.joblib")
        meta = {
            "cfg": asdict(self.cfg),
            "metrics": self.metrics,
            "feature_names": self.feature_names,
        }
        with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if hasattr(self, "decoded_df"):
            states_path = Path(out_dir) / "states.csv"
            self.decoded_df.to_csv(states_path, index=False)

    @staticmethod
    def load(model_dir: str) -> "RegimeHMMTrainer":
        meta_path = Path(model_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {model_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = RegimeHMMConfig(**meta["cfg"])
        obj = RegimeHMMTrainer(cfg)
        obj.metrics = meta.get("metrics", {})
        obj.feature_names = meta.get("feature_names", [])
        obj.model = joblib.load(Path(model_dir) / "model.joblib")
        obj.scaler = joblib.load(Path(model_dir) / "scaler.joblib")
        return obj
