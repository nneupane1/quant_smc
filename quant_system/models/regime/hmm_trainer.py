"""
Unsupervised regime trainer using Gaussian HMM on 6h/12h features.

Saves:
- model.joblib
- meta.json (config + feature list)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from hmmlearn.hmm import GaussianHMM
import joblib


@dataclass
class HMMConfig:
    n_states: int = 4
    covariance_type: str = "full"
    min_covar: float = 1e-3
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    verbose: bool = False


class HMMTrainer:
    def __init__(self, cfg: HMMConfig, features: List[str]):
        self.cfg = cfg
        self.features = features
        self.pipeline: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.features:
            raise ValueError("No features specified for HMM training.")

        X = df[self.features].astype(float).values

        model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            min_covar=self.cfg.min_covar,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            random_state=self.cfg.random_state,
            verbose=self.cfg.verbose,
        )

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("hmm", model),
            ]
        )

        pipe.fit(X)
        self.pipeline = pipe

        # quick state distribution
        hidden_states = pipe.named_steps["hmm"].predict(pipe.named_steps["scaler"].transform(X))
        state_counts = np.bincount(hidden_states, minlength=self.cfg.n_states).tolist()

        return {
            "n_states": self.cfg.n_states,
            "state_counts": state_counts,
        }

    def save(self, out_dir: str):
        if self.pipeline is None:
            raise RuntimeError("Call fit() before save().")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, Path(out_dir) / "model.joblib")
        meta = {"cfg": asdict(self.cfg), "features": self.features}
        with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(model_dir: str) -> "Pipeline":
        return joblib.load(Path(model_dir) / "model.joblib")
