"""Compatibility preprocessor wrapper for the canonical data prep layer."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.prep.preprocessing import Preprocessor as CanonicalPreprocessor
from quant_system.utils.logger import get_logger

LOG = get_logger("feature_preprocessor")


class FeaturePreprocessor:
    """
    Backward-compatible wrapper around the canonical preprocessing engine in
    `quant_system.data.prep.preprocessing`.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None, conf_dir: Optional[str] = None):
        if conf_dir is None:
            if config_loader is not None and getattr(config_loader, "conf_dir", None):
                conf_dir = str(config_loader.conf_dir)
            else:
                conf_dir = "quant_system/config"
        self.pre = CanonicalPreprocessor(conf_dir)
        LOG.info("FeaturePreprocessor initialized with canonical Preprocessor backend.")

    def apply(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Iterable[str]] = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Preprocess a feature dataframe. If `feature_cols` is provided, only those
        columns are returned after preprocessing.
        """
        out = self.pre.fit_transform(df) if fit else self.pre.transform(df)
        if feature_cols is not None:
            cols = [c for c in feature_cols if c in out.columns]
            return out[cols]
        return out


Preprocessor = FeaturePreprocessor
