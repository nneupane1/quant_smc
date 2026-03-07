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
import time
import json
from typing import Dict, List, Iterable, Optional
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

    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        pair: str = "BTCUSD",
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
    ):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.pair = pair
        self.resume = bool(resume)
        self.checkpoint_path = checkpoint_path or os.path.join(self.output_dir, f"{self.pair}_tf_checkpoint.json")

        log(f"TimeframeBuilder initialized for {input_csv}")

    # ------------------------------------------------------------
    def _setup_writers(self, *, append: bool) -> Dict[str, CSVWriter]:
        """Create CSV writers for each timeframe."""
        writers = {}
        for tf in self.TF_MAP:
            path = os.path.join(self.output_dir, f"{self.pair}_{tf}.csv")
            w = CSVWriter(path, append=append)
            if not append:
                w.write_header()
            writers[tf] = w
        return writers

    # ------------------------------------------------------------
    def _outputs_exist(self) -> bool:
        for tf in self.TF_MAP:
            path = os.path.join(self.output_dir, f"{self.pair}_{tf}.csv")
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
        return True

    # ------------------------------------------------------------
    def _load_checkpoint(self) -> Dict[str, object]:
        if not self.resume or not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return {}
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return {}
            if str(payload.get("pair")) != str(self.pair):
                return {}
            if str(payload.get("input_csv")) != str(self.input_csv):
                return {}
            if str(payload.get("output_dir")) != str(self.output_dir):
                return {}
            return payload
        except Exception as exc:
            log(f"Warning: failed to load timeframe checkpoint {self.checkpoint_path}: {exc}")
            return {}

    # ------------------------------------------------------------
    def _save_checkpoint(
        self,
        *,
        last_offset: int,
        total_rows: int,
        bars: Dict[str, Dict[str, float]],
        completed: bool,
    ) -> None:
        if not self.checkpoint_path:
            return
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        payload = {
            "pair": self.pair,
            "input_csv": self.input_csv,
            "output_dir": self.output_dir,
            "last_offset": int(last_offset),
            "total_rows": int(total_rows),
            "bars": bars,
            "completed": bool(completed),
            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------
    @staticmethod
    def _parse_row(line: str, header_cols: List[str]) -> Optional[Dict[str, str]]:
        raw = line.strip()
        if not raw:
            return None
        fields = next(csv.reader([raw]))
        if not fields or len(fields) < len(header_cols):
            return None
        return {k: fields[i] for i, k in enumerate(header_cols)}

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
    def build(self, chunk_size: int = 200000, include_final_partial: bool = False) -> None:
        """
        Main resample procedure:
        - Streams CSV in chunks
        - Resamples to 15m, 1h, 6h, 12h
        - Writes final bars upon window completion
        """

        t0 = time.time()
        log("Starting timeframe build (CSV-only).")
        os.makedirs(self.output_dir, exist_ok=True)

        file_size = os.path.getsize(self.input_csv) if os.path.exists(self.input_csv) else 0
        ckpt = self._load_checkpoint()
        resume_ready = False
        start_offset = 0
        total_rows = 0

        # Active bar per TF
        bars: Dict[str, Dict[str, float]] = {tf: None for tf in self.TF_MAP}
        if ckpt:
            ckpt_offset = int(ckpt.get("last_offset", 0) or 0)
            ckpt_rows = int(ckpt.get("total_rows", 0) or 0)
            ckpt_bars = ckpt.get("bars", {})
            if isinstance(ckpt_bars, dict):
                for tf in self.TF_MAP:
                    bar = ckpt_bars.get(tf)
                    bars[tf] = bar if isinstance(bar, dict) else None

            if bool(ckpt.get("completed")) and ckpt_offset >= file_size and self._outputs_exist():
                log("Timeframe outputs already current for this raw file. Skipping rebuild.")
                return

            if self._outputs_exist() and ckpt_offset > 0 and ckpt_offset <= file_size:
                start_offset = ckpt_offset
                total_rows = ckpt_rows
                resume_ready = True
                log(
                    f"Resuming timeframe build from checkpoint offset={start_offset:,} "
                    f"rows={total_rows:,}"
                )
            elif ckpt_offset > file_size:
                log("Checkpoint offset exceeded raw file size; restarting timeframe build from scratch.")

        writers = self._setup_writers(append=resume_ready)
        intervals = self.TF_MAP

        with open(self.input_csv, "r") as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError(f"Input CSV is empty: {self.input_csv}")
            header_cols = next(csv.reader([header_line.strip()]))
            data_start_offset = f.tell()
            if "timestamp" not in header_cols or "close" not in header_cols or "volume" not in header_cols:
                raise ValueError(
                    "Input CSV is missing required columns: timestamp, close, volume"
                )
            if start_offset < data_start_offset:
                start_offset = data_start_offset
            if start_offset > 0:
                f.seek(start_offset)

            chunk = []
            while True:
                line = f.readline()
                if not line:
                    break
                row = self._parse_row(line, header_cols)
                if row is None:
                    continue
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    self._process_chunk(chunk, bars, writers, intervals)
                    total_rows += len(chunk)
                    log(f"Processed {total_rows:,} rows.")
                    self._save_checkpoint(
                        last_offset=f.tell(),
                        total_rows=total_rows,
                        bars=bars,
                        completed=False,
                    )
                    chunk = []

            if chunk:
                self._process_chunk(chunk, bars, writers, intervals)
                total_rows += len(chunk)
                log(f"Processed final rows, total={total_rows:,}.")
                self._save_checkpoint(
                    last_offset=f.tell(),
                    total_rows=total_rows,
                    bars=bars,
                    completed=False,
                )

        if include_final_partial:
            log("Writing remaining bars at end of file.")
            for tf, bar in bars.items():
                if bar is not None:
                    self._write_bar(writers[tf], bar)
        else:
            log("Skipping remaining open bars at EOF to avoid partial HTF candles.")

        self._save_checkpoint(
            last_offset=file_size,
            total_rows=total_rows,
            bars=bars,
            completed=True,
        )
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
