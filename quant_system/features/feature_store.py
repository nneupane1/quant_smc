"""
Feature Store
-------------

A unified CSV-based feature store for the entire quant system.

Responsibilities:
    - Persist multi-timeframe aligned feature rows
    - Versioned, schema-aware storage
    - Deterministic loading, merging, slicing
    - Used by:
        * Label generator
        * ML trainer
        * Backtester
        * Forward tester
        * Regime modeling

Design:
    CSV-only storage for transparency and auditability.
    No pandas used; pure Python, scalable to multi-year datasets.

Storage format:
    root/
        15m_features_vX.csv
        1h_features_vX.csv
        6h_features_vX.csv
        12h_features_vX.csv
"""

import csv
import os
from typing import Dict, List, Any, Optional
from quant_system.utils.logger import log


class FeatureStore:
    """
    Manage writing, reading, merging, and slicing of TF feature CSVs.

    Parameters:
        root: directory for feature storage
        version: semantic or numeric version identifier
    """

    def __init__(self, root: str, version: str = "v1"):
        self.root = root
        self.version = version
        os.makedirs(self.root, exist_ok=True)
        log(f"FeatureStore initialized at {root} (version={version}).")

    def _path(self, tf: str) -> str:
        return os.path.join(self.root, f"{tf}_features_{self.version}.csv")

    def write_features(
        self,
        tf: str,
        features: Dict[int, Dict[str, Any]],
        overwrite: bool = False
    ) -> None:
        """
        Persist features to CSV:
            features: { ts → { col → val } }
        """

        path = self._path(tf)
        exists = os.path.exists(path)

        if exists and not overwrite:
            raise RuntimeError(f"File already exists: {path}")

        # Collect header
        if not features:
            log(f"No features provided for TF={tf}. Skipping write.")
            return

        sample_ts = next(iter(features.keys()))
        header = ["timestamp"] + list(features[sample_ts].keys())

        log(f"Writing {len(features):,} feature rows to {path}.")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ts, row in features.items():
                writer.writerow([ts] + [row[col] for col in header[1:]])

        log(f"Feature write complete: {path}")

    def append_features(
        self,
        tf: str,
        features: Dict[int, Dict[str, Any]],
    ) -> None:
        """
        Append new rows to an existing TF feature CSV.
        """

        path = self._path(tf)
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot append; file does not exist: {path}")

        log(f"Appending {len(features):,} rows to {path}.")

        # Read header
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            for ts, row in features.items():
                writer.writerow([ts] + [row.get(col, "") for col in header[1:]])

        log("Append complete.")

    def load_features(self, tf: str) -> Dict[int, Dict[str, Any]]:
        """
        Load TF feature CSV into memory as a dictionary.
        """

        path = self._path(tf)
        if not os.path.exists(path):
            raise RuntimeError(f"File not found: {path}")

        log(f"Loading features from {path}.")

        out: Dict[int, Dict[str, Any]] = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts = int(r["timestamp"])
                del r["timestamp"]
                out[ts] = {k: self._convert_value(v) for k, v in r.items()}

        log(f"Loaded {len(out):,} feature rows from {path}.")
        return out

    def merge_features(
        self,
        base: Dict[int, Dict[str, Any]],
        additional: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Merge two feature dictionaries by timestamp.
        """

        log("Merging feature dictionaries.")
        merged = {}

        keys = sorted(set(base.keys()) | set(additional.keys()))
        for ts in keys:
            row = {}
            if ts in base:
                row.update(base[ts])
            if ts in additional:
                row.update(additional[ts])
            merged[ts] = row

        log(f"Merged feature rows: {len(merged):,}")
        return merged

    def slice_features(
        self,
        tf: str,
        start_ts: int,
        end_ts: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Load only the rows within [start_ts, end_ts].
        """

        log(f"Slicing features for {tf}: {start_ts} → {end_ts}")
        path = self._path(tf)
        if not os.path.exists(path):
            raise RuntimeError(f"File not found: {path}")

        out = {}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts = int(r["timestamp"])
                if ts < start_ts or ts > end_ts:
                    continue
                del r["timestamp"]
                out[ts] = {k: self._convert_value(v) for k, v in r.items()}

        log(f"Sliced rows: {len(out):,}")
        return out

    def _convert_value(self, v: str) -> Any:
        """Convert CSV strings to numeric types when possible."""
        if v.strip() == "":
            return None
        try:
            if "." in v:
                return float(v)
            return int(v)
        except:
            return v
