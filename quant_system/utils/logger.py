import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Lightweight logger factory used across the codebase.
    Falls back to basicConfig once to avoid duplicate handlers.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            level=level,
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log(message: str, level: int = logging.INFO, logger: Optional[logging.Logger] = None):
    """
    Convenience wrapper for legacy log usage.
    """
    (logger or get_logger("quant_system")).log(level, message)
