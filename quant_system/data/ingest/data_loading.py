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
from typing import Dict, List, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.store.datamodel import Candle
from quant_system.data.prep.resampler import TimeframeResampler
from quant_system.utils.logger import get_logger

LOG = get_logger("data_loader")


class DataLoader:
    """
    Multi-asset CSV loader with resampling + rolling-window cleaning.
    """

    def __init__(self, config_loader: ConfigLoader, data_root: Optional[str] = None):
        self.config_loader = config_loader
        self.asset_cfg = config_loader.load_yaml("assets.yaml")
        self.assets_meta = self.asset_cfg["metadata"]
        self.active_asset = self.asset_cfg["default_asset"]
        storage_cfg = config_loader.load_yaml("storage.yaml").get("paths", {})

        self.data_root = data_root or storage_cfg.get("raw_1m", "data/raw_1m")
        self.tf_root = storage_cfg.get("tf", "data/tf")
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
    def _csv_candidates(self, asset: Optional[str] = None) -> List[str]:
        a = asset or self.active_asset
        pair = self.assets_meta.get(a, {}).get("kraken_pair")
        pair_candidates = []
        if pair:
            pair_candidates = [
                os.path.join(self.data_root, f"{pair}_1m.csv"),
                os.path.join("data/raw_1m", f"{pair}_1m.csv"),
                os.path.join("data/raw", f"{pair}_1m.csv"),
            ]
        return [
            os.path.join(self.data_root, f"{a}_1m.csv"),
            os.path.join(self.data_root, f"{a}_1m_kraken.csv"),
            os.path.join("data/raw_1m", f"{a}_1m.csv"),
            os.path.join("data/raw", f"{a}_1m.csv"),
            os.path.join("data/raw", f"{a}_1m_kraken.csv"),
            *pair_candidates,
        ]

    def _csv_path(self, asset: Optional[str] = None) -> str:
        for path in self._csv_candidates(asset):
            if os.path.exists(path):
                return path
        return self._csv_candidates(asset)[0]

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
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns for {a}: {sorted(missing)}")

        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"], utc=True)
        else:
            df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("dt").reset_index(drop=True)
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])

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

        df = df.replace([float("inf"), float("-inf")], pd.NA)
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

        df_15m = self._drop_incomplete_tail(df_1m, resampler.resample(df_1m, "15min"), "15min")
        df_1h = self._drop_incomplete_tail(df_1m, resampler.resample(df_1m, "1h"), "1h")
        df_6h = self._drop_incomplete_tail(df_1m, resampler.resample(df_1m, "6h"), "6h")
        df_12h = self._drop_incomplete_tail(df_1m, resampler.resample(df_1m, "12h"), "12h")

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

    def _drop_incomplete_tail(
        self,
        raw_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
        freq: str,
    ) -> pd.DataFrame:
        if raw_df.empty or resampled_df.empty:
            return resampled_df

        last_raw_dt = pd.to_datetime(raw_df["dt"], utc=True).max()
        last_bar_close = pd.to_datetime(resampled_df["dt"], utc=True).iloc[-1]
        if last_raw_dt < last_bar_close:
            trimmed = resampled_df.iloc[:-1].reset_index(drop=True)
            LOG.info(
                f"[DataLoader] Dropped partial {freq} tail bar "
                f"(last_raw={last_raw_dt}, bar_close={last_bar_close})"
            )
            return trimmed
        return resampled_df

    def write_timeframes(
        self,
        asset: Optional[str] = None,
        clean_window: int = 200,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Materialize the higher-timeframe CSVs from the canonical raw 1m file.
        """
        a = asset or self.active_asset
        out_dir = output_dir or self.tf_root
        os.makedirs(out_dir, exist_ok=True)

        dfs_tf = self.load_asset_all(a, clean_window=clean_window)
        written: Dict[str, str] = {}
        for tf, df in dfs_tf.items():
            if tf == "1m":
                continue
            path = os.path.join(out_dir, f"{a}_{tf}.csv")
            out_df = df.copy()
            if "dt" in out_df.columns:
                out_df["dt"] = pd.to_datetime(out_df["dt"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
            out_df.to_csv(path, index=False)
            written[tf] = path
            LOG.info(f"[DataLoader] Wrote {tf} CSV for {a} → {path}")
        return written
