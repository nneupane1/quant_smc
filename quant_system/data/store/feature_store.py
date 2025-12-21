"""
Unified feature store for all timeframes.
Handles saving and loading engineered feature tables (CSV-only, append-safe).
"""

import os
import pandas as pd
from typing import List, Optional
from pathlib import Path

from quant_system.config.config_manager import ConfigManager
from quant_system.utils.logger import get_logger

LOG = get_logger("feature_store")


class FeatureStore:
    """
    Persistent storage for all features.
    Uses CSV (not Parquet), partitioned by timeframe.
    Ensures schema consistency and provides safe updates.
    """

    def __init__(self, base_dir: str, conf_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        cfg = ConfigManager(conf_dir).get("features")
        self.join_cfg = cfg.get("join", {})
        self.scale_cfg = cfg.get("scale", {})

        LOG.info(f"FeatureStore initialized at {self.base}")

    # ----------------------------------------------------------
    # Internal utilities
    # ----------------------------------------------------------
    def _file(self, timeframe: str) -> Path:
        return self.base / f"features_{timeframe}.csv"

    def _validate_schema(self, df: pd.DataFrame, required_cols: List[str]):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"FeatureStore schema error. Missing columns: {missing}")

    # ----------------------------------------------------------
    # Save features
    # ----------------------------------------------------------
    def save(
        self,
        df: pd.DataFrame,
        timeframe: str,
        mode: str = "append",
        required_cols: Optional[List[str]] = None,
    ):
        """
        Save a feature dataframe for a given timeframe.
        mode = "append" or "overwrite".
        """
        f = self._file(timeframe)

        LOG.info(f"Saving features for {timeframe} in {mode} mode to {f}")

        if required_cols:
            self._validate_schema(df, required_cols)

        if mode == "overwrite" or not f.exists():
            df.to_csv(f, index=True)
            LOG.info(f"FeatureStore: overwritten {f} ({len(df)} rows).")
            return

        # append mode
        existing = pd.read_csv(f, index_col=0, parse_dates=True)
        merged = pd.concat([existing, df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.to_csv(f)

        LOG.info(f"FeatureStore: appended into {f}, total rows now {len(merged)}.")

    # ----------------------------------------------------------
    # Load features for training / forward run
    # ----------------------------------------------------------
    def load(self, timeframe: str) -> pd.DataFrame:
        """
        Load stored features for timeframe.
        Returns empty DataFrame if file does not exist.
        """
        f = self._file(timeframe)
        if not f.exists():
            LOG.info(f"FeatureStore: no file found for timeframe {timeframe}")
            return pd.DataFrame()

        df = pd.read_csv(f, index_col=0, parse_dates=True)
        LOG.info(f"FeatureStore: loaded {len(df)} rows from {f}")
        return df

    # ----------------------------------------------------------
    # Merge features into the training master frame
    # ----------------------------------------------------------
    def merge_into(
        self,
        base_df: pd.DataFrame,
        timeframe: str,
        how: str = "left",
        suffix: str = "",
    ) -> pd.DataFrame:
        """
        Merge feature table for timeframe into base_df using timestamp index.
        """
        fdf = self.load(timeframe)
        if fdf.empty:
            LOG.info(f"FeatureStore: merge skipped for {timeframe} (empty dataset)")
            return base_df

        df = base_df.join(fdf, how=how, rsuffix=suffix)
        LOG.info(f"FeatureStore: merged {timeframe} features into master frame")
        return df

    # ----------------------------------------------------------
    # Delete features for a given timeframe (rare ops)
    # ----------------------------------------------------------
    def drop(self, timeframe: str):
        f = self._file(timeframe)
        if f.exists():
            f.unlink()
            LOG.info(f"FeatureStore: deleted file {f}")
        else:
            LOG.info(f"FeatureStore: nothing to delete for {timeframe}")
