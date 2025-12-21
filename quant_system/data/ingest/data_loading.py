"""
data_loading.py — full multi-asset replacement

Responsibilities:
 - Load raw Kraken CSV per asset
 - Switch assets on demand
 - Produce rolling-window cleaned DataFrames
 - Integrate dynamic TF resampling (15m, 1h, 6h, 12h)
 - Expose unified interface for all downstream modules
"""

import os
import pandas as pd
from typing import Dict, Optional

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.data.prep.resampler import TimeframeResampler
from quant_system.utils.logger import get_logger

LOG = get_logger("data_loader")


class DataLoader:
    """
    Multi-asset CSV loader with resampling + rolling-window cleaning.
    """

    def __init__(self, config_loader: ConfigLoader, data_root: str = "data/raw"):
        self.config_loader = config_loader
        self.asset_cfg = config_loader.load_yaml("assets.yaml")
        self.assets_meta = self.asset_cfg["metadata"]
        self.active_asset = self.asset_cfg["default_asset"]

        self.data_root = data_root
        self.cache: Dict[str, pd.DataFrame] = {}

        LOG.info(f"[DataLoader] Initialized. Default asset={self.active_asset}")

    # ----------------------------------------------------------------------
    # SWITCH ACTIVE ASSET
    # ----------------------------------------------------------------------
    def set_asset(self, asset: str):
        if asset not in self.assets_meta:
            raise ValueError(f"Asset {asset} is not defined in assets.yaml")

        self.active_asset = asset
        LOG.info(f"[DataLoader] Asset switched → {asset}")

    # ----------------------------------------------------------------------
    # BUILD CSV PATH
    # ----------------------------------------------------------------------
    def _csv_path(self, asset: Optional[str] = None) -> str:
        a = asset or self.active_asset
        fname = f"{a}_1m.csv"
        return os.path.join(self.data_root, fname)

    # ----------------------------------------------------------------------
    # LOAD RAW 1M DATA
    # ----------------------------------------------------------------------
    def load_raw_1m(self, asset: Optional[str] = None) -> pd.DataFrame:
        a = asset or self.active_asset

        if a in self.cache:
            LOG.info(f"[DataLoader] Returning cached 1m data for {a}")
            return self.cache[a]

        fp = self._csv_path(a)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"CSV not found: {fp}")

        LOG.info(f"[DataLoader] Loading 1m CSV for {a} → {fp}")

        df = pd.read_csv(fp)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values("dt").reset_index(drop=True)

        self.cache[a] = df
        LOG.info(f"[DataLoader] Loaded {len(df)} rows for {a}")

        return df

    # ----------------------------------------------------------------------
    # ROLLING WINDOW CLEANING
    # ----------------------------------------------------------------------
    def apply_rolling_features(self, df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
        LOG.info(f"[DataLoader] Applying rolling-window cleaning → window={window}")

        df = df.copy()
        df["ret"] = df["close"].pct_change()
        df["ret_z"] = (df["ret"] - df["ret"].rolling(window).mean()) / df["ret"].rolling(window).std()

        df["vol"] = df["ret"].rolling(window).std()
        df["vol_z"] = (df["vol"] - df["vol"].rolling(window).mean()) / df["vol"].rolling(window).std()

        df = df.dropna().reset_index(drop=True)
        LOG.info(f"[DataLoader] Rolling-window features applied. Remaining rows={len(df)}")

        return df

    # ----------------------------------------------------------------------
    # RESAMPLE: 15m / 1h / 6h / 12h
    # ----------------------------------------------------------------------
    def resample_all(self, df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict of TF DataFrames:
         {
           "15m": df_15m,
           "1h":  df_1h,
           "6h":  df_6h,
           "12h": df_12h
         }
        """
        LOG.info("[DataLoader] Starting resampling (15m / 1h / 6h / 12h)")

        resampler = TimeframeResampler()

        df_15m = resampler.resample(df_1m, "15min")
        df_1h  = resampler.resample(df_1m, "1h")
        df_6h  = resampler.resample(df_1m, "6h")
        df_12h = resampler.resample(df_1m, "12h")

        LOG.info("[DataLoader] Resampling complete")

        return {
            "15m": df_15m,
            "1h": df_1h,
            "6h": df_6h,
            "12h": df_12h
        }

    # ----------------------------------------------------------------------
    # LOAD EVERYTHING FOR ONE ASSET
    # ----------------------------------------------------------------------
    def load_asset_all(self, asset: Optional[str] = None, clean_window: int = 200):
        """
        Full load pipeline for any asset:
         1) Load raw 1m CSV
         2) Apply rolling features
         3) Resample into execution TFs
        """
        a = asset or self.active_asset
        LOG.info(f"[DataLoader] Full load pipeline for asset={a}")

        df_raw = self.load_raw_1m(a)
        df_clean = self.apply_rolling_features(df_raw, window=clean_window)
        dfs_tf = self.resample_all(df_clean)

        return {
            "1m": df_clean,
            **dfs_tf
        }
