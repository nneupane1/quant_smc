"""
Preprocessing utilities:
Scaling, winsorization, NaN handling, walk-forward splits.
Config-driven and used by training + forward-test pipelines.
"""

import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from quant_system.config.config_manager import ConfigManager
from quant_system.utils.logger import get_logger

LOG = get_logger("preprocessor")


class Preprocessor:
    """
    Config-driven preprocessing:
        - NaN handling
        - ffill limits
        - Winsorization
        - Scaling (standard | minmax)
        - Walk-forward split
    """

    def __init__(self, conf_dir: str):
        cfg = ConfigManager(conf_dir).get("features")

        self.join_cfg = cfg.get("join", {})
        self.scale_cfg = cfg.get("scale", {})
        self.clip_pct = float(self.scale_cfg.get("clip_pct", 0.995))

        scale_type = self.scale_cfg.get("type", "standard")
        if scale_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.interpolate_missing = bool(self.join_cfg.get("interpolate_missing", False))
        self.ffill_limit = int(self.join_cfg.get("ffill_limit", 1))
        self.drop_incomplete = bool(self.join_cfg.get("drop_incomplete_rows", True))

        LOG.info(f"Preprocessor initialized: scale={scale_type}, clip_pct={self.clip_pct}")

    # --------------------------------------------------------------
    # NaN handling
    # --------------------------------------------------------------
    def _clean_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.interpolate_missing:
            df = df.interpolate(limit=self.ffill_limit)

        df = df.ffill(limit=self.ffill_limit)

        if self.drop_incomplete:
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                LOG.info(f"Preprocessor: dropped {before - after} incomplete rows.")

        return df

    # --------------------------------------------------------------
    # Winsorization
    # --------------------------------------------------------------
    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        numeric_cols = out.select_dtypes(include=["number", "bool"]).columns
        if len(numeric_cols) == 0:
            return out
        lower = out[numeric_cols].quantile(1 - self.clip_pct)
        upper = out[numeric_cols].quantile(self.clip_pct)
        out[numeric_cols] = out[numeric_cols].clip(lower=lower, upper=upper, axis=1)
        return out

    # --------------------------------------------------------------
    # Scaling (fit-transform for train; transform only for test)
    # --------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        LOG.info("Preprocessing: fit-transform start")

        df = self._clean_nans(df)
        df = self._winsorize(df)

        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        if not numeric_cols:
            return df

        out = df.copy()
        arr = self.scaler.fit_transform(out[numeric_cols].values)
        out.loc[:, numeric_cols] = arr

        LOG.info("Preprocessing: fit-transform complete")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        LOG.info("Preprocessing: transform start")

        df = self._clean_nans(df)
        df = self._winsorize(df)

        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        if not numeric_cols:
            return df

        out = df.copy()
        arr = self.scaler.transform(out[numeric_cols].values)
        out.loc[:, numeric_cols] = arr

        LOG.info("Preprocessing: transform complete")
        return out

    # --------------------------------------------------------------
    # Walk-forward split
    # --------------------------------------------------------------
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_until: str,
        valid_until: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train and validation chronologically.
        train_until: last timestamp for training (inclusive).
        valid_until: optional final timestamp for validation.
        """

        LOG.info(f"Walk-forward split: train_until={train_until}, valid_until={valid_until}")

        train_df = df.loc[:train_until]

        if valid_until:
            valid_df = df.loc[train_until:valid_until]
        else:
            valid_df = df.loc[train_until:]

        LOG.info(f"Walk-forward: train={len(train_df)} rows, valid={len(valid_df)} rows")
        return train_df, valid_df

    # --------------------------------------------------------------
    # Feature filtering (optional)
    # --------------------------------------------------------------
    def select_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        LOG.info(f"Preprocessor: selecting {len(cols)} features")
        return df[cols]
