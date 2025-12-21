"""
Data Ingestion Engine
---------------------

Responsible for orchestrating:
- Historical OHLCV downloads from KrakenClient
- Writing raw 1-minute data to CSV
- Validating ranges and detecting gaps
- Logging detailed progress for transparency

This file produces clean, append-friendly CSVs:
./data/raw/BTCUSD_1m.csv
"""

import os
import csv
import time
from typing import List, Dict, Any
from datetime import datetime

from quant_system.data.ingest.kraken_client import KrakenClient
from quant_system.utils.logger import log
from quant_system.data.store.writer import CSVWriter
from quant_system.data.store.datamodel import Candle


class DataIngestion:
    """
    Download multi-year historical 1-minute data from Kraken,
    store in high-quality CSV files, and validate dataset completeness.
    """

    def __init__(
        self,
        pair: str,
        start_ts: int,
        end_ts: int,
        output_path: str,
        batch_sleep: float = 0.5
    ):
        self.client = KrakenClient(pair=pair)
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.output_path = output_path
        self.batch_sleep = batch_sleep

        log(f"DataIngestion initialized for {pair} from {start_ts} to {end_ts}")
        log(f"Saving output to: {self.output_path}")

    # ------------------------------------------------------------
    def _ensure_dir(self) -> None:
        """Ensure output directory exists."""
        dirname = os.path.dirname(self.output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            log(f"Created directory: {dirname}")

    # ------------------------------------------------------------
    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[Candle]:
        """Convert normalized dict rows → strongly typed Candle datamodel."""
        return [
            Candle(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
            for row in rows
        ]

    # ------------------------------------------------------------
    def _validate(self, rows: List[Candle]) -> None:
        """
        Basic integrity checks:
        - Monotonic timestamps
        - No duplicate timestamps
        - Timestamp boundaries respected
        """
        log("Validation started.")

        timestamps = [r.timestamp for r in rows]

        if timestamps != sorted(timestamps):
            raise ValueError("Timestamps are not strictly monotonic increasing.")

        if len(timestamps) != len(set(timestamps)):
            raise ValueError("Duplicate timestamps detected in downloaded data.")

        if timestamps[0] > self.start_ts:
            log("Warning: Earliest timestamp is later than expected start.")

        if timestamps[-1] < self.end_ts:
            log("Warning: Latest timestamp is earlier than expected end.")

        minutes = (timestamps[-1] - timestamps[0]) // 60
        log(f"Time span covers about {minutes:,} minutes.")

        log("Validation complete.")

    # ------------------------------------------------------------
    def run(self) -> None:
        """
        Execute ingestion:
        - Fetch all 1m data from Kraken
        - Normalize & validate rows
        - Write to CSV
        """

        self._ensure_dir()

        t0 = time.time()
        log("Starting ingestion run.")

        raw_rows = self.client.fetch_ohlcv(
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            batch_sleep=self.batch_sleep
        )

        log(f"Fetched {len(raw_rows):,} rows from Kraken. Normalizing...")

        candles = self._normalize_rows(raw_rows)

        log("Validating dataset integrity.")
        self._validate(candles)

        writer = CSVWriter(self.output_path)
        writer.write_candles(candles)

        dt = time.time() - t0
        log(f"Ingestion completed in {dt:.2f} seconds.")
