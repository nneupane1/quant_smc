from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Iterable, TypeVar

from quant_system.utils.logger import get_logger


F = TypeVar("F", bound=Callable)


def timed(logger_name: str = "quant_system", level: int = logging.INFO) -> Callable[[F], F]:
    """
    Decorator that logs function runtime without changing return behavior.
    """

    def outer(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            logger = get_logger(logger_name)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.log(level, "%s completed in %.4fs", fn.__qualname__, elapsed)

        return inner  # type: ignore[return-value]

    return outer


def log_exceptions(
    logger_name: str = "quant_system",
    *,
    level: int = logging.ERROR,
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that logs an exception and optionally re-raises it.
    """

    def outer(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.log(level, "%s failed: %s", fn.__qualname__, exc, exc_info=True)
                if reraise:
                    raise
                return None

        return inner  # type: ignore[return-value]

    return outer


def requires_columns(columns: Iterable[str]) -> Callable[[F], F]:
    """
    Decorator for dataframe-style methods expecting a `df` positional argument.
    Raises a clear error when required columns are missing.
    """

    required = tuple(columns)

    def outer(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            df = kwargs.get("df")
            if df is None and len(args) >= 2:
                df = args[1]
            if df is None or not hasattr(df, "columns"):
                raise TypeError(f"{fn.__qualname__} requires a dataframe-like argument with columns.")

            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"{fn.__qualname__} missing required columns: {missing}")
            return fn(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return outer
