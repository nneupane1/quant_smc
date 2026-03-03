"""
Legacy compatibility surface for regime HMM training.

This module preserves the older `quant_system.models.regime.hmm_trainer`
API used by CLI utilities, while delegating to the canonical implementation in
`quant_system.ml.training.regime_hmm_trainer`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_system.ml.training.regime_hmm_trainer import (
    RegimeHMMConfig as _CanonicalConfig,
    RegimeHMMTrainer as _CanonicalTrainer,
)


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
        self.features = list(features or [])
        self._trainer = _CanonicalTrainer(
            _CanonicalConfig(
                n_states=cfg.n_states,
                covariance_type=cfg.covariance_type,
                seed=cfg.random_state,
                feature_cols=self.features or None,
            )
        )
        self.pipeline = None
        self._report: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        states = self._trainer.fit_transform(df)
        counts = states["state"].value_counts().sort_index().to_dict() if "state" in states.columns else {}
        self._report = {
            "n_states": int(self.cfg.n_states),
            "state_counts": {int(k): int(v) for k, v in counts.items()},
            "rows": int(len(states)),
            "features": list(self._trainer.features_),
        }
        self.pipeline = self._trainer
        return self._report

    def save(self, out_dir: str):
        self._trainer.save(out_dir)

    @staticmethod
    def load(model_dir: str) -> _CanonicalTrainer:
        return _CanonicalTrainer.load(model_dir)
