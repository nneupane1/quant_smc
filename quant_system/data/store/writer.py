"""
CSV Writer Utilities
--------------------

Provides high-performance CSV writing utilities used by the
data ingestion pipeline and by all transformed timeframes.

Features:
- Header-aware writing
- Overwrite and append modes
- Safe flushing for multi-million row datasets
- Detailed console logs for every write operation
"""

import os
import csv
from typing import Iterable

from .datamodel import Candle
from quant_system.utils.logger import log


class CSVWriter:
    """
    A robust CSV writer for OHLCV and derived candle types.
    Ensures consistent column order and safe writes.
    """

    COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, path: str, append: bool = False):
        self.path = path
        self.append = append
        self._mode = "a" if append else "w"

        log(f"CSVWriter initialized — path={self.path}, append={self.append}")

    # ------------------------------------------------------------
    def _ensure_dir(self) -> None:
        """Ensure parent directory exists."""
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            log(f"Created directory: {directory}")

    # ------------------------------------------------------------
    def write_header(self) -> None:
        """Write CSV header if file is new or overwrite requested."""
        self._ensure_dir()

        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)

        log("CSV header written.")

    # ------------------------------------------------------------
    def write_candles(self, candles: Iterable[Candle]) -> None:
        """
        Write iterable of Candle objects to CSV.
        Includes header automatically if file does not exist.
        """

        self._ensure_dir()

        header_needed = not os.path.exists(self.path) or not self.append

        with open(self.path, self._mode, newline="") as f:
            writer = csv.writer(f)

            if header_needed:
                writer.writerow(self.COLUMNS)

            rows_written = 0
            for c in candles:
                writer.writerow([
                    c.timestamp,
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume
                ])
                rows_written += 1

        log(f"Finished writing {rows_written:,} rows to {self.path}")
