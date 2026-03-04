"""Kraken OHLC downloader with retry, compatibility fetches, and CSV export."""

import csv
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from quant_system.data.ingest.api_retry import RetrySession
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import console_stage, fmt_num, fmt_seconds, fmt_ts, get_logger

LOG = get_logger("kraken_client")

KRAKEN_BASE = "https://api.kraken.com/0/public/OHLC"


class KrakenClient:
    """
    Kraken OHLC downloader with multi-asset switching and detailed logs.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.load_yaml("assets.yaml")
        self.asset = self.config["default_asset"]
        self.meta = self.config["metadata"][self.asset]

        self.api_key = os.getenv("KRAKEN_API_KEY", "")
        self.api_secret = os.getenv("KRAKEN_API_SECRET", "")

        self.session = RetrySession()

        LOG.info(f"[KrakenClient] Initialized | default asset: {self.asset} ({self.meta['kraken_pair']})")

    # ----------------------------------------------------------------------
    # ASSET SELECTION
    # ----------------------------------------------------------------------
    def set_asset(self, asset: str):
        if asset not in self.config["metadata"]:
            raise ValueError(f"Asset {asset} not configured in assets.yaml")

        self.asset = asset
        self.meta = self.config["metadata"][asset]

        LOG.info(f"[KrakenClient] Asset switched -> {asset} (kraken={self.meta['kraken_pair']})")

    # ----------------------------------------------------------------------
    # INTERNAL OHLC FETCH
    # ----------------------------------------------------------------------
    def _fetch_chunk(self, interval: int, since: int) -> Optional[tuple]:
        """
        Fetch one OHLC chunk. Kraken returns:
        { "result": { "<pair>": [ [ts, open, high, low, close, v, ...], ... ], "last": <id> } }
        """
        params = {
            "pair": self.meta["kraken_pair"],
            "interval": interval,
            "since": since
        }

        try:
            r = self.session.get(KRAKEN_BASE, params=params, timeout=20)
            j = r.json()

            if "error" in j and j["error"]:
                LOG.error(f"[KrakenClient] OHLC error -> {j['error']}")
                return None

            result = j.get("result", {})
            # filter out the "last" key
            pair_keys = [k for k in result.keys() if k != "last"]
            if not pair_keys:
                return None
            pair_key = pair_keys[0]
            last_id = int(result.get("last", 0))
            return result[pair_key], last_id

        except Exception as e:
            LOG.error(f"[KrakenClient] Exception fetching chunk -> {e}")
            return None

    def _rows_to_dicts(
        self,
        chunk: List[List[Any]],
        end_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in chunk:
            try:
                ts = int(float(row[0]))
                if end_ts is not None and ts > end_ts:
                    continue
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[6]),  # Kraken index 6 = volume
                    }
                )
            except Exception as e:
                LOG.error(f"[KrakenClient] Failed mapping row -> {e}")
        return rows

    def fetch_ohlcv(
        self,
        start_ts: int,
        end_ts: int,
        batch_sleep: float = 1.2,
        interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Compatibility fetch API used by the ingestion layer.
        Returns normalized dict rows:
            {timestamp, open, high, low, close, volume}
        """
        LOG.info(
            f"[KrakenClient] Fetch OHLCV {self.asset} | interval={interval}m | "
            f"{start_ts} -> {end_ts}"
        )

        rows: List[Dict[str, Any]] = []
        seen_ts = set()
        since = start_ts
        last_progress_ts = start_ts
        stale_steps = 0
        request_count = 0
        t0 = time.time()

        while since <= end_ts:
            request_count += 1
            fetched = self._fetch_chunk(interval, since)
            if not fetched:
                console_stage(
                    "Kraken fetch stalled",
                    f"request={request_count} cursor={since} received no data",
                    status="warn",
                )
                break

            raw_rows, last_id = fetched
            chunk_rows = self._rows_to_dicts(raw_rows, end_ts=end_ts)
            new_rows = 0

            for row in chunk_rows:
                ts = row["timestamp"]
                if ts < start_ts or ts in seen_ts:
                    continue
                seen_ts.add(ts)
                rows.append(row)
                new_rows += 1
                last_progress_ts = ts

            if new_rows == 0:
                stale_steps += 1
                if stale_steps > 3:
                    console_stage(
                        "Kraken fetch stopped",
                        f"request={request_count} repeated empty progress at {fmt_ts(last_progress_ts)}",
                        status="warn",
                    )
                    break
            else:
                stale_steps = 0

            latest_chunk_ts = chunk_rows[-1]["timestamp"] if chunk_rows else last_progress_ts
            console_stage(
                f"Kraken request {request_count}",
                (
                    f"new={fmt_num(new_rows)} total={fmt_num(len(rows))} "
                    f"latest={fmt_ts(latest_chunk_ts)} next_cursor={int(last_id or 0)}"
                ),
                status="info",
            )

            next_since = int(last_id or 0)
            if next_since <= since:
                next_since = last_progress_ts + interval * 60
            if next_since <= since:
                console_stage(
                    "Kraken fetch stopped",
                    f"cursor did not advance after {fmt_ts(last_progress_ts)}",
                    status="warn",
                )
                break

            since = next_since
            if last_progress_ts >= end_ts:
                break
            time.sleep(batch_sleep)

        console_stage(
            "Kraken fetch complete",
            (
                f"requests={request_count} rows={fmt_num(len(rows))} "
                f"start={fmt_ts(start_ts)} end={fmt_ts(last_progress_ts if rows else start_ts)} "
                f"elapsed={fmt_seconds(time.time() - t0)}"
            ),
            status="ok",
        )
        return rows

    # ----------------------------------------------------------------------
    # PUBLIC METHOD: DOWNLOAD FULL HISTORY
    # ----------------------------------------------------------------------
    def download_history_csv(
        self,
        interval: int,
        output_csv: str,
        start_year: int = 2017,
        end_year: int = 2026
    ):
        """
        Download full history from Kraken (year by year),
        write to CSV using Candle datamodel.
        """

        LOG.info(f"[KrakenClient] Starting full-history download for {self.asset}")
        LOG.info(f"               interval={interval}m  {start_year} -> {end_year}")
        LOG.info(f"               output: {output_csv}")

        # Ensure output directory exists
        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            LOG.info(f"[KrakenClient] Created directory: {out_dir}")

        header_written = False
        columns = ["dt", "timestamp", "open", "high", "low", "close", "volume"]

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            for year in range(start_year, end_year + 1):
                start_ts = int(datetime(year, 1, 1).timestamp())
                year_end = int(datetime(year + 1, 1, 1).timestamp()) - interval * 60
                LOG.info(f"[KrakenClient] Fetching year {year}, range={start_ts} -> {year_end}")

                candles = self.fetch_ohlcv(
                    start_ts=start_ts,
                    end_ts=year_end,
                    batch_sleep=1.2,
                    interval=interval,
                )
                if not candles:
                    LOG.info("[KrakenClient] Empty year result, continuing.")
                    continue

                if not header_written:
                    writer.writerow(columns)
                    header_written = True

                for row in candles:
                    ts = row["timestamp"]
                    writer.writerow(
                        [
                            datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                            ts,
                            row["open"],
                            row["high"],
                            row["low"],
                            row["close"],
                            row["volume"],
                        ]
                    )

                LOG.info(
                    f"[KrakenClient]   Wrote {len(candles)} candles | "
                    f"last ts={candles[-1]['timestamp']}"
                )

                LOG.info(f"[KrakenClient] Completed year {year}")

        LOG.info(f"[KrakenClient] DONE | History stored at: {output_csv}")

    # ----------------------------------------------------------------------
    # RANGE-BASED DOWNLOAD (supports resume/append)
    # ----------------------------------------------------------------------
    def download_range_csv(
        self,
        start_ts: int,
        end_ts: int,
        interval: int,
        output_csv: str,
        append: bool = False,
    ) -> Tuple[int, int]:
        """
        Download OHLCV between start_ts and end_ts (inclusive, unix seconds).
        Writes CSV with columns: dt (UTC, minute precision), timestamp, ohlcv.
        Returns (rows_written, last_timestamp).
        """
        LOG.info(
            f"[KrakenClient] Range download {self.asset} | interval={interval}m | "
            f"{start_ts} -> {end_ts} | output={output_csv} | append={append}"
        )

        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            LOG.info(f"[KrakenClient] Created directory: {out_dir}")

        rows = self.fetch_ohlcv(
            start_ts=start_ts,
            end_ts=end_ts,
            batch_sleep=1.2,
            interval=interval,
        )

        mode = "a" if append and os.path.exists(output_csv) else "w"
        header_needed = not (append and os.path.exists(output_csv))
        rows_written = 0
        last_ts = start_ts

        with open(output_csv, mode, newline="") as f:
            writer = csv.writer(f)
            columns = ["dt", "timestamp", "open", "high", "low", "close", "volume"]
            if header_needed:
                writer.writerow(columns)

            for row in rows:
                ts = row["timestamp"]
                writer.writerow(
                    [
                        datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                        ts,
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                    ]
                )
                rows_written += 1
                last_ts = ts

        LOG.info(f"[KrakenClient] DONE | rows={rows_written:,} | last_ts={last_ts}")
        return rows_written, last_ts

    # ----------------------------------------------------------------------
    # MAP RAW KRAKEN CANDLES -> tuples for CSV
    # ----------------------------------------------------------------------
    def _map_chunk(self, chunk: List[List], end_ts: Optional[int] = None) -> List[tuple]:
        return [
            (
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            )
            for row in self._rows_to_dicts(chunk, end_ts=end_ts)
        ]
