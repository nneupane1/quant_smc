"""
Helpers for leak-safe forward targets used by NARX forecasters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_targets(df15: pd.DataFrame, horizons: tuple[int, ...] = (4, 8, 16)) -> pd.DataFrame:
    """
    Add forward log-return targets to a 15m feature frame.

    Each target is: log(close_{t+H} / close_t) using closed bars only.
    """
    df = df15.copy()
    df = df.sort_values("dt").reset_index(drop=True)

    close = df["close"].astype(float)
    for h in horizons:
        df[f"ret_fwd_{h}"] = np.log(close.shift(-h) / close)
    return df
