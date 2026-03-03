"""Compatibility wrapper over the canonical dataframe-based feature store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant_system.data.store.feature_store import FeatureStore as CanonicalFeatureStore
from quant_system.utils.logger import get_logger

LOG = get_logger("legacy_feature_store")


class FeatureStore(CanonicalFeatureStore):
    """
    Legacy `quant_system.features.feature_store.FeatureStore` API preserved on top
    of the canonical dataframe-based store in `quant_system.data.store`.
    """

    def __init__(self, root: Optional[str] = None, version: str = "v1", conf_dir: str = "quant_system/config"):
        self.version = version
        base_dir = Path(root) if root else None
        super().__init__(base_dir=str(base_dir) if base_dir else None, conf_dir=conf_dir)
        self.root = str(self.base)
        LOG.info("Legacy FeatureStore wrapper initialized at %s (version=%s).", self.root, version)

    def _path(self, tf: str) -> str:
        return str(self._file(tf))

    @staticmethod
    def _dict_to_frame(features: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        if not features:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(features, orient="index")
        df.index = pd.to_datetime(df.index.astype("int64"), unit="s", utc=True)
        df.index.name = "dt"
        return df.sort_index()

    @staticmethod
    def _frame_to_dict(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        if df is None or df.empty:
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        ts_index = (pd.to_datetime(df.index, utc=True).astype("int64") // 10**9).astype(int)
        clean = df.where(pd.notna(df), None)
        for ts, (_, row) in zip(ts_index.tolist(), clean.iterrows()):
            out[int(ts)] = row.to_dict()
        return out

    def write_features(self, tf: str, features: Dict[int, Dict[str, Any]], overwrite: bool = False) -> None:
        frame = self._dict_to_frame(features)
        if frame.empty:
            return
        mode = "overwrite" if overwrite else "append"
        self.save(frame, tf, mode=mode)

    def append_features(self, tf: str, features: Dict[int, Dict[str, Any]]) -> None:
        frame = self._dict_to_frame(features)
        if frame.empty:
            return
        self.save(frame, tf, mode="append")

    def load_features(self, tf: str) -> Dict[int, Dict[str, Any]]:
        return self._frame_to_dict(self.load(tf))

    def merge_features(
        self,
        base: Dict[int, Dict[str, Any]],
        additional: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        base_frame = self._dict_to_frame(base)
        add_frame = self._dict_to_frame(additional)
        if base_frame.empty:
            return self._frame_to_dict(add_frame)
        if add_frame.empty:
            return self._frame_to_dict(base_frame)
        merged = base_frame.join(add_frame, how="outer", rsuffix="_dup")
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dup in dup_cols:
            base_col = dup[:-4]
            merged[base_col] = merged[dup].combine_first(merged.get(base_col))
        merged = merged.drop(columns=dup_cols)
        return self._frame_to_dict(merged.sort_index())

    def slice_features(self, tf: str, start_ts: int, end_ts: int) -> Dict[int, Dict[str, Any]]:
        frame = self.load(tf)
        if frame.empty:
            return {}
        lo = pd.to_datetime(int(start_ts), unit="s", utc=True)
        hi = pd.to_datetime(int(end_ts), unit="s", utc=True)
        sliced = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
        return self._frame_to_dict(sliced)
