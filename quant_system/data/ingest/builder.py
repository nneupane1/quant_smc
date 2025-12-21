"""
Timeframe Builder (CSV-only)
----------------------------

A high-performance CSV resampling engine that transforms raw 1-minute
historical data into higher timeframes: 15m, 1h, 6h, 12h.

Features:
- Streaming load of extremely large CSV datasets.
- Strict non-repainting resample ([start, end) windowing).
- Precise bar-close rule (15m closes at HH:15/30/45/00 etc).
- Produces clean, consistent, header-aware CSV files.
- Detailed transparent console logging.
"""

import os
import csv
import math
import time
from typing import Dict, List, Iterable
from datetime import datetime

from quant_system.utils.logger import log
from quant_system.data.store.datamodel import Candle, TFCandleBatch
from quant_system.data.store.writer import CSVWriter


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def floor_ts(ts: int, interval_sec: int) -> int:
    """Return the floored timestamp to the start of the TF window."""
    return (ts // interval_sec) * interval_sec


def ts_to_str(ts: int) -> str:
    """Readable datetime for logging."""
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------
# Timeframe Builder
# ------------------------------------------------------------
class TimeframeBuilder:
    """
    Resample a large 1-minute CSV dataset into:
    - 15m
    - 1h
    - 6h
    - 12h

    The builder outputs CSV files:
      data/tf/BTCUSD_15m.csv
      data/tf/BTCUSD_1h.csv
      data/tf/BTCUSD_6h.csv
      data/tf/BTCUSD_12h.csv
    """

    TF_MAP = {
        "15m": 60 * 15,
        "1h":  60 * 60,
        "6h":  60 * 60 * 6,
        "12h": 60 * 60 * 12,
    }

    def __init__(self, input_csv: str, output_dir: str, pair: str = "BTCUSD"):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.pair = pair

        log(f"TimeframeBuilder initialized for {input_csv}")

    # ------------------------------------------------------------
    def _setup_writers(self) -> Dict[str, CSVWriter]:
        """Create CSV writers for each timeframe."""
        writers = {}
        for tf in self.TF_MAP:
            path = os.path.join(self.output_dir, f"{self.pair}_{tf}.csv")
            w = CSVWriter(path, append=False)
            w.write_header()
            writers[tf] = w
        return writers

    # ------------------------------------------------------------
    def _new_bar(self, ts: int, price: float, volume: float) -> Dict[str, float]:
        """Initialize a new bar dict."""
        return {
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
        }

    # ------------------------------------------------------------
    def _update_bar(self, bar: Dict[str, float], price: float, volume: float) -> None:
        """Update OHLCV fields for the current window."""
        if price > bar["high"]:
            bar["high"] = price
        if price < bar["low"]:
            bar["low"] = price
        bar["close"] = price
        bar["volume"] += volume

    # ------------------------------------------------------------
    def _write_bar(self, writer: CSVWriter, bar: Dict[str, float]) -> None:
        """Convert dict → Candle → write to CSV."""
        candle = Candle(
            timestamp=bar["timestamp"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"]
        )
        writer.write_candles([candle])

    # ------------------------------------------------------------
    def build(self, chunk_size: int = 200000) -> None:
        """
        Main resample procedure:
        - Streams CSV in chunks
        - Resamples to 15m, 1h, 6h, 12h
        - Writes final bars upon window completion
        """

        t0 = time.time()
        log("Starting timeframe build (CSV-only).")

        writers = self._setup_writers()

        # Active bar per TF
        bars: Dict[str, Dict[str, float]] = {tf: None for tf in self.TF_MAP}
        intervals = self.TF_MAP

        total_rows = 0

        with open(self.input_csv, "r") as f:
            reader = csv.DictReader(f)

            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    self._process_chunk(chunk, bars, writers, intervals)
                    total_rows += len(chunk)
                    log(f"Processed {total_rows:,} rows.")
                    chunk = []

            if chunk:
                self._process_chunk(chunk, bars, writers, intervals)
                total_rows += len(chunk)
                log(f"Processed final rows, total={total_rows:,}.")

        # Write remaining open bars
        log("Writing remaining bars at end of file.")
        for tf, bar in bars.items():
            if bar is not None:
                self._write_bar(writers[tf], bar)

        dt = time.time() - t0
        log(f"Timeframe build completed in {dt:.2f} seconds.")

    # ------------------------------------------------------------
    def _process_chunk(
        self,
        chunk: List[Dict[str, str]],
        bars: Dict[str, Dict[str, float]],
        writers: Dict[str, CSVWriter],
        intervals: Dict[str, int]
    ) -> None:
        """
        Process a chunk of 1m rows, updating and closing bars
        across all target timeframes.
        """

        for r in chunk:
            ts = int(r["timestamp"])
            price = float(r["close"])
            volume = float(r["volume"])

            for tf, interval in intervals.items():

                window_start = floor_ts(ts, interval)

                # Open a new bar if none exists
                if bars[tf] is None:
                    bars[tf] = self._new_bar(window_start, price, volume)
                    continue

                # If timestamp stays inside the bar window, update it
                if window_start == bars[tf]["timestamp"]:
                    self._update_bar(bars[tf], price, volume)
                    continue

                # Otherwise close bar and open a new one
                self._write_bar(writers[tf], bars[tf])
                bars[tf] = self._new_bar(window_start, price, volume)

    # ------------------------------------------------------------
    @staticmethod
    def estimate_bar_counts(
        start_year: int = 2017,
        end_year: int = 2026
    ) -> Dict[str, int]:
        """
        Estimate bar counts per TF for planning resource usage.
        """

        years = end_year - start_year
        minutes = years * 365 * 24 * 60

        estimates = {
            "1m": minutes,
            "15m": minutes // 15,
            "1h": minutes // 60,
            "6h": minutes // (6 * 60),
            "12h": minutes // (12 * 60),
        }

        return estimates
