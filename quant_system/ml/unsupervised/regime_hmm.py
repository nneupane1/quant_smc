"""
Regime HMM trainer (unsupervised, 6h/12h).

- Fits a Gaussian HMM on simple market descriptors (returns, range%, volume z-score).
- Outputs state IDs per bar and persists the model + metadata.

Leak safety: uses only past bars; no future lookahead. Designed for right-closed bars
on 6h/12h timeframes (but works with any evenly spaced TF that has dt/open/high/low/close/volume).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeHMMConfig:
    n_states: int = 5
    covariance_type: str = "full"  # {"full","diag","spherical","tied"}
    n_iter: int = 200
    random_seed: int = 42
    feature_cols: Optional[List[str]] = None  # if None, build default descriptors


class RegimeHMMTrainer:
    def __init__(self, cfg: RegimeHMMConfig):
        self.cfg = cfg
        self.model: Optional[GaussianHMM] = None
        self.features_used: List[str] = []

    def _build_default_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.sort_values("dt").reset_index(drop=True)
        # basic descriptors
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        vol = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)

        ret = np.log(close).diff().fillna(0.0)
        range_pct = (high - low) / close.replace(0, np.nan)
        range_pct = range_pct.fillna(0.0)
        vol_z = (vol - vol.rolling(64, min_periods=16).mean()) / (
            vol.rolling(64, min_periods=16).std().replace(0, np.nan)
        )
        vol_z = vol_z.fillna(0.0)

        feat = pd.DataFrame(
            {
                "ret": ret,
                "range_pct": range_pct,
                "vol_z": vol_z,
            }
        )
        return feat, feat.columns.tolist()

    def _select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if self.cfg.feature_cols:
            cols = [c for c in self.cfg.feature_cols if c in df.columns]
            if not cols:
                feat, names = self._build_default_features(df)
            else:
                feat = df[cols].copy()
                feat = feat.sort_values("dt").reset_index(drop=True)
                names = cols
        else:
            feat, names = self._build_default_features(df)
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feat, names

    def fit(self, df: pd.DataFrame) -> pd.Series:
        """
        Fit HMM and return state IDs aligned to df rows.
        Requires columns: dt, open, high, low, close (volume optional).
        """
        df = df.copy()
        if "dt" not in df.columns:
            if "timestamp" in df.columns:
                df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            else:
                raise ValueError("Input must have dt or timestamp column.")

        feat, names = self._select_features(df)
        self.features_used = names

        X = feat.values.astype(float)
        model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.random_seed,
            verbose=False,
        )
        model.fit(X)
        self.model = model

        states = model.predict(X)
        df_states = pd.Series(states, name="regime_state")
        return df_states

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        joblib.dump(self.model, out_dir / "regime_hmm.joblib")
        meta = {
            "config": asdict(self.cfg),
            "features": self.features_used,
        }
        (out_dir / "meta.json").write_text(pd.io.json.dumps(meta, indent=2))

    @staticmethod
    def load(model_dir: Path) -> "RegimeHMMTrainer":
        meta = pd.read_json(model_dir / "meta.json")
        cfg = RegimeHMMConfig(**meta["config"])
        obj = RegimeHMMTrainer(cfg)
        obj.features_used = meta["features"]
        obj.model = joblib.load(model_dir / "regime_hmm.joblib")
        return obj
