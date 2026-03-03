"""Unified CSV feature store with canonical datetime-index handling."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from quant_system.config.config_manager import ConfigManager
from quant_system.utils.logger import get_logger

LOG = get_logger("feature_store")


class FeatureStore:
    """
    Persistent storage for all features.
    Uses CSV (not Parquet), partitioned by timeframe.
    Ensures schema consistency and provides safe updates.
    """

    def __init__(self, base_dir: Optional[str] = None, conf_dir: str = "quant_system/config"):
        manager = ConfigManager(conf_dir)
        storage_cfg = manager.full.get("paths", {})
        base_path = base_dir or storage_cfg.get("features", "data/features")
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

        cfg = manager.get("features")
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

    def _normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "dt" in out.columns:
                out["dt"] = pd.to_datetime(out["dt"], utc=True)
                out = out.set_index("dt")
            elif "timestamp" in out.columns:
                out.index = pd.to_datetime(out["timestamp"], unit="s", utc=True)
            else:
                raise ValueError("FeatureStore requires a DatetimeIndex or `dt` / `timestamp` column.")
        elif out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")

        out = out.sort_index()
        out.index.name = "dt"
        return out

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

        df = self._normalize_frame(df)
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
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df.index.name = "dt"
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

        base_norm = self._normalize_frame(base_df)
        df = base_norm.join(fdf, how=how, rsuffix=suffix)
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
