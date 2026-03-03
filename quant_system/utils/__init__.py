from .decorators import log_exceptions, requires_columns, timed
from .logger import get_logger, log
from .rolling import (
    crossed_above,
    crossed_below,
    ewm_slope,
    rolling_percentile,
    rolling_zscore,
    safe_shift,
)

__all__ = [
    "SessionClassifier",
    "crossed_above",
    "crossed_below",
    "ewm_slope",
    "get_logger",
    "log",
    "log_exceptions",
    "requires_columns",
    "rolling_percentile",
    "rolling_zscore",
    "safe_shift",
    "timed",
]


def __getattr__(name):
    if name == "SessionClassifier":
        from .time_utils import SessionClassifier

        return SessionClassifier
    raise AttributeError(f"module 'quant_system.utils' has no attribute {name!r}")
