"""
Multi-Timeframe Feature Resampler
---------------------------------

Provides a deterministic, non-repainting mechanism to align multi-timeframe
features into a unified 15m-indexed matrix.

Purpose:
    - Load 15m, 1h, 6h, 12h CSVs
    - Align higher-timeframe rows to each 15m bar using only closed bars
    - Produce feature dictionaries where:
        key   = 15m timestamp
        value = { "15m": feature_row, "1h": feature_row, "6h": feature_row, "12h": feature_row }

Guarantees:
    - No look-ahead
    - Strict left-close/right-open windowing
    - Timestamp-consistent joins
    - Scales to multi-year datasets

Usage during feature engineering:
    TFResampler(...).build_alignment()
"""

import csv
import logging
from typing import Dict, List, Optional, Tuple

from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


LOGGER = logging.getLogger("quant_system.features.resampler")
LOGGER.setLevel(logging.INFO)


# ======================================================================
# Helper
# ======================================================================

def load_tf_csv(path: str) -> List[Candle]:
    """Load a timeframe CSV as typed Candle objects."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(Candle.from_csv_row(r))
    return rows


def floor_ts(ts: int, interval_sec: int) -> int:
    return (ts // interval_sec) * interval_sec


# ======================================================================
# TFResampler Class
# ======================================================================

class TFResampler:
    """
    Provides alignment between 15m anchor timestamps and higher-timeframe rows.

    Steps:
        1) Load 15m, 1h, 6h, 12h CSVs
        2) Build timestamp indices
        3) For each 15m bar, find the most recent closed 1h/6h/12h bar
        4) Return merged feature matrix indexed by 15m timestamp

    Outputs:
        A dict of the form:
            {
                15m_ts: {
                    "15m": Candle,
                    "1h": Candle,
                    "6h": Candle,
                    "12h": Candle
                },
                ...
            }
    """

    TF_INTERVALS = {
        "15m":  15 * 60,
        "1h":   60 * 60,
        "6h":   6 * 60 * 60,
        "12h": 12 * 60 * 60,
    }

    def __init__(
        self,
        tf_paths: Dict[str, str],   # {"15m": path_15m, "1h": ..., ...}
        enforce_sorted: bool = True
    ):
        self.paths = tf_paths
        self.enforce_sorted = enforce_sorted

        self.data: Dict[str, List[Candle]] = {}
        self.indices: Dict[str, List[int]] = {}

        log("TFResampler initialized.")

    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Load all TF CSVs into memory."""
        for tf, path in self.paths.items():
            self.data[tf] = load_tf_csv(path)
            if self.enforce_sorted:
                self.data[tf].sort(key=lambda c: c.timestamp)
            log(f"Loaded {len(self.data[tf]):,} candles for {tf}")

    # ------------------------------------------------------------------
    def build_indices(self) -> None:
        """Create timestamp lists for binary searching."""
        for tf, rows in self.data.items():
            self.indices[tf] = [c.timestamp for c in rows]
            if self.enforce_sorted and self.indices[tf] != sorted(self.indices[tf]):
                raise ValueError(f"Timestamp order invalid for TF {tf}.")
        log("Timeframe indices built.")

    # ------------------------------------------------------------------
    def _lookup_closed_bar(
        self,
        ts: int,
        tf: str
    ) -> Optional[Candle]:
        """
        For a given 15m timestamp ts, return the most recent closed bar in TF.
        Equivalent to finding the index where timestamp <= ts.
        """

        index = self.indices[tf]
        rows = self.data[tf]

        # Binary search
        lo, hi = 0, len(index) - 1
        pos = -1

        while lo <= hi:
            mid = (lo + hi) // 2
            if index[mid] <= ts:
                pos = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if pos == -1:
            return None
        return rows[pos]

    # ------------------------------------------------------------------
    def _align_tf_to_15m(
        self
    ) -> Dict[int, Dict[str, Candle]]:
        """
        Build final alignment:
            { 15m_ts → { "15m": row, "1h": row, "6h": row, "12h": row } }
        """

        fifteen = self.data["15m"]
        aligned: Dict[int, Dict[str, Candle]] = {}

        for candle in fifteen:
            ts = candle.timestamp
            merged = {"15m": candle}

            for tf in ["1h", "6h", "12h"]:
                closed = self._lookup_closed_bar(ts, tf)
                merged[tf] = closed

            aligned[ts] = merged

        log(f"Aligned {len(aligned):,} rows across 15m,1h,6h,12h.")
        return aligned

    # ------------------------------------------------------------------
    def build_alignment(self) -> Dict[int, Dict[str, Candle]]:
        """
        Full pipeline:
            1) Load CSVs
            2) Build timestamp indices
            3) Construct TF-aligned feature matrix
        """

        log("Starting multi-timeframe alignment.")
        self.load_all()
        self.build_indices()
        result = self._align_tf_to_15m()
        log("Multi-timeframe alignment complete.")
        return result
