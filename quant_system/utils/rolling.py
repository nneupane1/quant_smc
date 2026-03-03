from __future__ import annotations

from typing import Union

import pandas as pd


SeriesLike = Union[pd.Series, pd.DataFrame]


def safe_shift(obj: SeriesLike, periods: int = 1, fill_value=None) -> SeriesLike:
    out = obj.shift(periods)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods if min_periods is not None else window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, pd.NA)
    return ((series - mean) / std).fillna(0.0)


def rolling_percentile(series: pd.Series, window: int, q: float, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods if min_periods is not None else window
    return series.rolling(window, min_periods=min_periods).quantile(q)


def ewm_slope(series: pd.Series, span: int = 10) -> pd.Series:
    ema = series.ewm(span=span, adjust=False).mean()
    return ema.diff().fillna(0.0)


def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left > right) & (left.shift(1) <= right.shift(1))).fillna(False)


def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left < right) & (left.shift(1) >= right.shift(1))).fillna(False)
