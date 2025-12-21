"""
HMM regime trainer for 6h/12h bars.

Produces latent state ids and probabilities, saves model + meta, and can
export a states CSV for downstream feature joins.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import json

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class HMMRegimeConfig:
    n_states: int = 5
    covariance_type: str = "full"
    seed: int = 42
    tf: str = "6h"  # or "12h"
    feature_cols: Optional[List[str]] = None
    eps: float = 1e-9
    vol_window: int = 14


class HMMRegimeTrainer:
    """
    Fits a Gaussian HMM on 6h/12h bars using simple returns/vol/dispersion features.
    """

    def __init__(self, cfg: Optional[HMMRegimeConfig] = None):
        self.cfg = cfg or HMMRegimeConfig()
        self.model: Optional[GaussianHMM] = None
        self.features_: Optional[List[str]] = None

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("dt")
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)

        logret = np.log(close / close.shift(1))
        absret = logret.abs()
        vol = logret.rolling(self.cfg.vol_window, min_periods=5).std()
        range_pct = (high - low) / close.replace(0, np.nan)
        dispersion = (close - open_) / (high - low + self.cfg.eps)

        feats = pd.DataFrame(
            {
                "logret": logret,
                "absret": absret,
                "vol": vol,
                "range_pct": range_pct,
                "dispersion": dispersion,
            }
        )
        feats["dt"] = df["dt"]
        feats = feats.dropna().reset_index(drop=True)
        return feats

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = self._build_features(df)
        X = feats.drop(columns=["dt"]).values

        self.model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            random_state=self.cfg.seed,
            n_iter=200,
        )
        self.model.fit(X)

        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        prob_cols = [f"state_prob_{i}" for i in range(self.cfg.n_states)]

        out = feats[["dt"]].copy()
        out["state_id"] = states
        prob_df = pd.DataFrame(probs, columns=prob_cols)
        out = pd.concat([out.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

        self.features_ = feats.columns.drop("dt").tolist()
        return out

    def save(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        meta = {
            "config": asdict(self.cfg),
            "features": self.features_,
        }
        with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.model is not None:
            joblib.dump(self.model, Path(out_dir) / "model.joblib")

    @staticmethod
    def load(model_dir: str) -> "HMMRegimeTrainer":
        with open(Path(model_dir) / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = HMMRegimeConfig(**meta["config"])
        trainer = HMMRegimeTrainer(cfg)
        trainer.features_ = meta.get("features")
        trainer.model = joblib.load(Path(model_dir) / "model.joblib")
        return trainer

    def predict_states(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted/loaded.")
        feats = self._build_features(df)
        X = feats.drop(columns=["dt"]).values
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        prob_cols = [f"state_prob_{i}" for i in range(self.model.n_components)]

        out = feats[["dt"]].copy()
        out["state_id"] = states
        prob_df = pd.DataFrame(probs, columns=prob_cols)
        out = pd.concat([out.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
        return out

    def build_live_row(self, df_tail: pd.DataFrame) -> pd.DataFrame:
        feats = self._build_features(df_tail)
        if feats.empty:
            raise ValueError("Not enough data to build features.")
        X = feats.drop(columns=["dt"]).iloc[[-1]]
        return X

    def predict_live(self, df_tail: pd.DataFrame) -> Dict[str, float]:
        X = self.build_live_row(df_tail)
        state = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]
        return {
            "state_id": state,
            **{f"state_prob_{i}": float(p) for i, p in enumerate(probs)},
        }
