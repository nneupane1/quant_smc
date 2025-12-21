"""
Rolling-window utilities used across the entire system:
    - lag features
    - rolling statistics
    - rolling z-scores
    - endogenous/exogenous windows
    - construction of ML-ready matrices (X, y) with no leakage
    - timestamp-preserving behavior until final removal
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from quant_system.utils.logger import log


class RollingWindows:
    """
    Central rolling-window engine.
    This module is used by:
        - FeatureBuilder
        - LabelLoader
        - ModelTrainer
        - Predictor
        - BacktestEngine
        - ForwardEngine
    """

    # ---------------------------------------------------------
    # Lags
    # ---------------------------------------------------------
    @staticmethod
    def add_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lag features for each column.
        """
        out = df.copy()
        for col in cols:
            for L in lags:
                new_col = f"{col}_lag{L}"
                out[new_col] = df[col].shift(L)
        log(f"RollingWindows: added {len(cols)*len(lags)} lagged columns.")
        return out

    # ---------------------------------------------------------
    # Rolling mean / std / min / max
    # ---------------------------------------------------------
    @staticmethod
    def add_rolling_stats(
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Add rolling mean, std, min, max for each column.
        """
        out = df.copy()
        for col in cols:
            for w in windows:
                out[f"{col}_rmean{w}"] = df[col].rolling(w).mean()
                out[f"{col}_rstd{w}"] = df[col].rolling(w).std()
                out[f"{col}_rmin{w}"] = df[col].rolling(w).min()
                out[f"{col}_rmax{w}"] = df[col].rolling(w).max()
        log(f"RollingWindows: added rolling stats for {len(cols)} cols and {len(windows)} windows.")
        return out

    # ---------------------------------------------------------
    # Rolling z-scores
    # ---------------------------------------------------------
    @staticmethod
    def add_rolling_zscore(
        df: pd.DataFrame,
        cols: List[str],
        window: int
    ) -> pd.DataFrame:
        """
        z = (x - mean) / std using rolling windows.
        """
        out = df.copy()
        for col in cols:
            mean = df[col].rolling(window).mean()
            std = df[col].rolling(window).std()
            out[f"{col}_z{window}"] = (df[col] - mean) / std.replace(0, np.nan)
        log(f"RollingWindows: added z-scores for {len(cols)} cols, window={window}.")
        return out

    # ---------------------------------------------------------
    # Rolling normalization band (for EMA stretch analysis)
    # ---------------------------------------------------------
    @staticmethod
    def add_band_regime(
        df: pd.DataFrame,
        col: str,
        upper: pd.Series,
        lower: pd.Series
    ) -> pd.DataFrame:
        """
        Create boolean features for band regime:
            inside, above, below.
        """
        out = df.copy()
        out[f"{col}_inband"] = ((df[col] >= lower) & (df[col] <= upper)).astype(int)
        out[f"{col}_aboveband"] = (df[col] > upper).astype(int)
        out[f"{col}_belowband"] = (df[col] < lower).astype(int)
        log(f"RollingWindows: band regime features created for {col}")
        return out

    # ---------------------------------------------------------
    # Endogenous windowing (model uses past `window` rows only)
    # ---------------------------------------------------------
    @staticmethod
    def build_endogenous_matrix(
        df: pd.DataFrame,
        feature_cols: List[str],
        window: int
    ) -> pd.DataFrame:
        """
        Keep only rows where at least `window` historical observations exist.
        """
        before = len(df)
        out = df.copy()
        out = out.iloc[window:]
        log(f"RollingWindows: endogenous window={window}, removed {before - len(out)} rows.")
        return out[feature_cols]

    # ---------------------------------------------------------
    # Exogenous windowing (multiple time series/features)
    # ---------------------------------------------------------
    @staticmethod
    def build_exogenous_matrix(
        df: pd.DataFrame,
        exo_cols: List[str],
        window: int
    ) -> pd.DataFrame:
        """
        Same logic as endogenous, but for externally supplied features.
        """
        before = len(df)
        out = df.copy()
        out = out.iloc[window:]
        log(f"RollingWindows: exogenous window={window}, removed {before - len(out)} rows.")
        return out[exo_cols]

    # ---------------------------------------------------------
    # Concise ML matrix builder
    # ---------------------------------------------------------
    @staticmethod
    def build_matrix(
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str],
        min_history: int,
        future_horizon: int
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Build ML-ready matrix ensuring:
            - no look-ahead (drop rows near the end)
            - enough past history (drop rows near the beginning)
            - timestamp preserved until final step
        """

        before = len(df)
        out = df.copy()

        # 1) ensure past window
        out = out.iloc[min_history:]

        # 2) ensure future label horizon
        if future_horizon > 0:
            out = out.iloc[:-future_horizon]

        log(
            f"RollingWindows: ML matrix built. "
            f"Removed {before - len(out)} rows (history={min_history}, horizon={future_horizon})."
        )

        if label_col:
            y = out[label_col].copy()
        else:
            y = None

        X = out[feature_cols].copy()
        return X, y

    # ---------------------------------------------------------
    # Final timestamp removal for ML
    # ---------------------------------------------------------
    @staticmethod
    def remove_timestamp(df: pd.DataFrame) -> np.ndarray:
        """
        Return a pure numeric matrix for ML, discarding index.
        """
        mat = df.to_numpy()
        log("RollingWindows: timestamp removed, matrix is ML-ready.")
        return mat
