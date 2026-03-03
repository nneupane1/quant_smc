"""Authoritative raw 1m ingestion orchestration for the data layer."""

import os
import time
from typing import Any, Dict, List, Optional

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.builder import TimeframeBuilder
from quant_system.data.ingest.kraken_client import KrakenClient
from quant_system.data.store.datamodel import Candle
from quant_system.data.store.writer import CSVWriter
from quant_system.utils.logger import log


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
        output_path: Optional[str] = None,
        tf_output_dir: Optional[str] = None,
        batch_sleep: float = 1.2,
        interval: int = 1,
        conf_dir: str = "quant_system/config",
        build_timeframes: bool = True,
    ):
        self.config_loader = ConfigLoader(conf_dir=conf_dir)
        self.asset_cfg = self.config_loader.load_yaml("assets.yaml")
        self.storage_cfg = self.config_loader.load_yaml("storage.yaml").get("paths", {})
        self.asset = self._resolve_asset_key(pair)

        self.client = KrakenClient(config_loader=self.config_loader)
        self.client.set_asset(self.asset)
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.interval = interval
        self.output_path = output_path or os.path.join(
            self.storage_cfg.get("raw_1m", "data/raw_1m"),
            f"{self.asset}_1m.csv",
        )
        self.tf_output_dir = tf_output_dir or self.storage_cfg.get("tf", "data/tf")
        self.batch_sleep = batch_sleep
        self.build_timeframes = build_timeframes

        log(f"DataIngestion initialized for {self.asset} from {start_ts} to {end_ts}")
        log(f"Saving output to: {self.output_path}")
        if self.build_timeframes:
            log(f"Timeframe output dir: {self.tf_output_dir}")

    def _resolve_asset_key(self, asset_or_pair: str) -> str:
        if asset_or_pair in self.asset_cfg["metadata"]:
            return asset_or_pair

        for asset, meta in self.asset_cfg["metadata"].items():
            if meta.get("kraken_pair") == asset_or_pair:
                return asset

        raise ValueError(f"Unknown asset or Kraken pair: {asset_or_pair}")

    # ------------------------------------------------------------
    def _ensure_dir(self) -> None:
        """Ensure output directory exists."""
        dirname = os.path.dirname(self.output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            log(f"Created directory: {dirname}")
        if self.build_timeframes and self.tf_output_dir and not os.path.exists(self.tf_output_dir):
            os.makedirs(self.tf_output_dir, exist_ok=True)
            log(f"Created timeframe directory: {self.tf_output_dir}")

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
    def run(self) -> Dict[str, Any]:
        """
        Execute ingestion:
        - Fetch all 1m data from Kraken
        - Normalize & validate rows
        - Write raw 1m CSV
        - Optionally build 15m / 1h / 6h / 12h CSVs
        """

        self._ensure_dir()

        t0 = time.time()
        log("Starting ingestion run.")

        raw_rows = self.client.fetch_ohlcv(
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            batch_sleep=self.batch_sleep,
            interval=self.interval,
        )

        log(f"Fetched {len(raw_rows):,} rows from Kraken. Normalizing...")

        candles = self._normalize_rows(raw_rows)
        if not candles:
            raise ValueError("No candles returned for requested window.")

        log("Validating dataset integrity.")
        self._validate(candles)

        writer = CSVWriter(self.output_path, append=False)
        writer.write_candles(candles)

        tf_paths: Dict[str, str] = {}
        if self.build_timeframes:
            builder = TimeframeBuilder(
                input_csv=self.output_path,
                output_dir=self.tf_output_dir,
                pair=self.asset,
            )
            builder.build()
            tf_paths = {
                tf: os.path.join(self.tf_output_dir, f"{self.asset}_{tf}.csv")
                for tf in TimeframeBuilder.TF_MAP
            }

        dt = time.time() - t0
        log(f"Ingestion completed in {dt:.2f} seconds.")
        return {
            "asset": self.asset,
            "rows": len(candles),
            "raw_1m_path": self.output_path,
            "tf_paths": tf_paths,
            "duration_sec": dt,
        }
