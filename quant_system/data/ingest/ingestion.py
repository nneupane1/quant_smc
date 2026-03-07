"""Authoritative raw 1m ingestion orchestration for the data layer."""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.builder import TimeframeBuilder
from quant_system.data.ingest.kraken_client import KrakenClient
from quant_system.data.store.datamodel import Candle
from quant_system.data.store.writer import CSVWriter
from quant_system.utils.logger import console_kv, console_stage, fmt_num, fmt_seconds, fmt_ts, log


@dataclass
class IngestionCheckpointState:
    asset: str
    pair: str
    interval: int
    requested_start_ts: int
    requested_end_ts: int
    raw_1m_path: str
    tf_output_dir: str
    last_processed_ts: Optional[int] = None
    resumed_from_ts: Optional[int] = None
    rows_written_last_run: int = 0
    total_rows_written: int = 0
    build_timeframes: bool = True
    completed: bool = False
    updated_at: str = ""


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
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
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
        self.resume = resume
        self.checkpoint_path = checkpoint_path or self._default_checkpoint_path()

        log(f"DataIngestion initialized for {self.asset} from {start_ts} to {end_ts}")
        log(f"Saving output to: {self.output_path}")
        if self.build_timeframes:
            log(f"Timeframe output dir: {self.tf_output_dir}")
        if self.resume:
            log(f"Checkpoint path: {self.checkpoint_path}")

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

    def _default_checkpoint_path(self) -> str:
        base, _ext = os.path.splitext(self.output_path)
        return f"{base}_checkpoint.json"

    def _load_checkpoint(self) -> Optional[IngestionCheckpointState]:
        if not self.resume or not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            state = IngestionCheckpointState(**payload)
            if state.asset != self.asset or state.interval != self.interval:
                return None
            return state
        except Exception as exc:
            log(f"Warning: failed to load checkpoint {self.checkpoint_path}: {exc}")
            return None

    def _save_checkpoint(
        self,
        *,
        last_processed_ts: Optional[int],
        resumed_from_ts: Optional[int],
        rows_written_last_run: int,
        total_rows_written: int,
        completed: bool,
    ) -> None:
        if not self.checkpoint_path:
            return
        state = IngestionCheckpointState(
            asset=self.asset,
            pair=self.asset_cfg["metadata"][self.asset]["kraken_pair"],
            interval=self.interval,
            requested_start_ts=self.start_ts,
            requested_end_ts=self.end_ts,
            raw_1m_path=self.output_path,
            tf_output_dir=self.tf_output_dir,
            last_processed_ts=last_processed_ts,
            resumed_from_ts=resumed_from_ts,
            rows_written_last_run=rows_written_last_run,
            total_rows_written=total_rows_written,
            build_timeframes=self.build_timeframes,
            completed=completed,
            updated_at=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        )
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)

    @staticmethod
    def _read_last_timestamp_from_csv(path: str) -> Optional[int]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                if end <= 0:
                    return None
                pos = end - 1
                while pos > 0:
                    f.seek(pos)
                    if f.read(1) == b"\n":
                        break
                    pos -= 1
                if pos == 0:
                    f.seek(0)
                line = f.readline().decode("utf-8").strip()
            if not line or line.startswith("timestamp"):
                return None
            return int(float(line.split(",", 1)[0]))
        except Exception as exc:
            log(f"Warning: failed to infer last timestamp from {path}: {exc}")
            return None

    def _resolve_resume_state(self) -> Dict[str, Any]:
        checkpoint = self._load_checkpoint()
        resumed_from_ts = checkpoint.last_processed_ts if checkpoint and checkpoint.last_processed_ts else None
        total_rows_written = checkpoint.total_rows_written if checkpoint else 0

        csv_last_ts = self._read_last_timestamp_from_csv(self.output_path)
        if csv_last_ts is not None:
            resumed_from_ts = max(filter(lambda x: x is not None, [resumed_from_ts, csv_last_ts]))

        if resumed_from_ts is None:
            return {
                "resume_start_ts": self.start_ts,
                "resumed_from_ts": None,
                "append": False,
                "total_rows_written": total_rows_written,
            }

        next_start = max(self.start_ts, int(resumed_from_ts) + self.interval * 60)
        return {
            "resume_start_ts": next_start,
            "resumed_from_ts": int(resumed_from_ts),
            "append": os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0,
            "total_rows_written": total_rows_written,
        }

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

        resume_state = self._resolve_resume_state()
        resume_start_ts = int(resume_state["resume_start_ts"])
        resumed_from_ts = resume_state["resumed_from_ts"]
        append_mode = bool(resume_state["append"])
        total_rows_written = int(resume_state["total_rows_written"] or 0)

        console_kv(
            "Data Run",
            {
                "asset": self.asset,
                "pair": self.asset_cfg["metadata"][self.asset]["kraken_pair"],
                "window_start": fmt_ts(self.start_ts),
                "window_end": fmt_ts(self.end_ts),
                "raw_1m": self.output_path,
                "tf_dir": self.tf_output_dir if self.build_timeframes else "-",
                "resume": self.resume,
            },
        )
        if resumed_from_ts is not None:
            log(f"Resuming ingestion from last processed ts={resumed_from_ts} -> next start={resume_start_ts}")
            console_stage(
                "Resume detected",
                f"last_processed={fmt_ts(resumed_from_ts)} next_start={fmt_ts(resume_start_ts)}",
                status="info",
            )

        if resume_start_ts > self.end_ts:
            log("Requested end already covered by existing raw file/checkpoint; skipping fetch.")
            console_stage(
                "Data already current",
                f"checkpoint/raw data already cover target end {fmt_ts(self.end_ts)}",
                status="ok",
            )
            candles: List[Candle] = []
            raw_rows: List[Dict[str, Any]] = []
        else:
            console_stage(
                "Kraken OHLC fetch",
                f"start={fmt_ts(resume_start_ts)} end={fmt_ts(self.end_ts)} interval={self.interval}m",
                status="info",
            )
            raw_rows = self.client.fetch_ohlcv(
                start_ts=resume_start_ts,
                end_ts=self.end_ts,
                batch_sleep=self.batch_sleep,
                interval=self.interval,
            )

            log(f"Fetched {len(raw_rows):,} rows from Kraken. Normalizing...")

            candles = self._normalize_rows(raw_rows)
            if resumed_from_ts is not None:
                candles = [c for c in candles if c.timestamp > resumed_from_ts]

            if candles:
                log("Validating dataset integrity.")
                self._validate(candles)

        rows_written = len(candles)
        if candles:
            writer = CSVWriter(self.output_path, append=append_mode)
            writer.write_candles(candles)
            total_rows_written += rows_written
            console_stage(
                "Raw 1m stored",
                f"new_rows={fmt_num(rows_written)} total_rows={fmt_num(total_rows_written)} path={self.output_path}",
                status="ok",
            )
        elif not os.path.exists(self.output_path):
            raise ValueError("No candles returned for requested window and no existing raw CSV found.")

        tf_paths: Dict[str, str] = {}
        tf_checkpoint_path: Optional[str] = None
        if self.build_timeframes:
            console_stage("Timeframe rebuild", f"source={self.output_path} -> {self.tf_output_dir}", status="info")
            tf_checkpoint_path = os.path.join(self.tf_output_dir, f"{self.asset}_tf_checkpoint.json")
            builder = TimeframeBuilder(
                input_csv=self.output_path,
                output_dir=self.tf_output_dir,
                pair=self.asset,
                checkpoint_path=tf_checkpoint_path,
                resume=self.resume,
            )
            builder.build()
            tf_paths = {
                tf: os.path.join(self.tf_output_dir, f"{self.asset}_{tf}.csv")
                for tf in TimeframeBuilder.TF_MAP
            }
            console_stage(
                "Timeframes ready",
                ", ".join(f"{tf}={path}" for tf, path in tf_paths.items()),
                status="ok",
            )

        dt = time.time() - t0
        last_processed_ts = candles[-1].timestamp if candles else resumed_from_ts
        completed = bool(last_processed_ts is not None and last_processed_ts >= self.end_ts)
        self._save_checkpoint(
            last_processed_ts=last_processed_ts,
            resumed_from_ts=resumed_from_ts,
            rows_written_last_run=rows_written,
            total_rows_written=total_rows_written,
            completed=completed,
        )
        log(f"Ingestion completed in {dt:.2f} seconds.")
        console_stage(
            "Data orchestration complete",
            (
                f"asset={self.asset} new_rows={fmt_num(rows_written)} "
                f"last_processed={fmt_ts(last_processed_ts)} elapsed={fmt_seconds(dt)}"
            ),
            status="ok",
        )
        return {
            "asset": self.asset,
            "rows": rows_written,
            "last_processed_ts": last_processed_ts,
            "resumed_from_ts": resumed_from_ts,
            "checkpoint_path": self.checkpoint_path,
            "tf_checkpoint_path": tf_checkpoint_path,
            "completed": completed,
            "raw_1m_path": self.output_path,
            "tf_paths": tf_paths,
            "duration_sec": dt,
        }
