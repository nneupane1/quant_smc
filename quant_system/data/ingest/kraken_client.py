"""
Kraken OHLC downloader with retry and CSV export.
"""

import os
import time
import csv
import requests
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from quant_system.data.ingest.api_retry import RetrySession
from quant_system.data.store.data_model import Candle
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger, log

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
        columns = ["timestamp", "open", "high", "low", "close", "volume"]

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            for year in range(start_year, end_year + 1):

                since = int(datetime(year, 1, 1).timestamp())
                LOG.info(f"[KrakenClient] Fetching year {year}, since={since}")

                while True:
                    chunk = self._fetch_chunk(interval, since)
                    if not chunk:
                        LOG.info("[KrakenClient] Empty chunk received, stopping year.")
                        break

                    candles = self._map_chunk(chunk)

                    # Write header only once
                    if not header_written:
                        writer.writerow(columns)
                        header_written = True

                    for c in candles:
                        writer.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])

                    LOG.info(f"[KrakenClient]   Wrote {len(candles)} candles | last ts={candles[-1].timestamp}")

                    since = candles[-1].timestamp

                    # Kraken limit protections
                    time.sleep(1.2)

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

        mode = "a" if append and os.path.exists(output_csv) else "w"
        header_needed = not (append and os.path.exists(output_csv))

        rows_written = 0
        last_ts = start_ts
        seen_ts = set()
        no_progress = 0
        prev_since = None

        with open(output_csv, mode, newline="") as f:
            writer = csv.writer(f)
            columns = ["dt", "timestamp", "open", "high", "low", "close", "volume"]
            if header_needed:
                writer.writerow(columns)

            since = start_ts
            while since <= end_ts:
                fetched = self._fetch_chunk(interval, since)
                if not fetched:
                    LOG.info("[KrakenClient] Empty chunk received, stopping range fetch.")
                    break

                raw_rows, last_id = fetched
                candles = self._map_chunk(raw_rows, end_ts=end_ts)
                if not candles:
                    no_progress += 1
                    if no_progress > 3:
                        LOG.info("[KrakenClient] No progress after multiple attempts; stopping.")
                        break
                    break
                no_progress = 0

                for ts, o, h, l, c, v in candles:
                    if ts in seen_ts:
                        continue
                    seen_ts.add(ts)
                    writer.writerow([
                        datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                        ts, o, h, l, c, v
                    ])
                    rows_written += 1
                    last_ts = ts

                LOG.info(f"[KrakenClient]   Wrote {len(candles)} candles | last ts={last_ts}")

                # Advance cursor using Kraken-provided last cursor
                if last_id == prev_since:
                    LOG.info("[KrakenClient] last cursor repeated; stopping to avoid loop.")
                    break
                prev_since = last_id
                since = last_id
                if since <= 0:
                    since = last_ts + interval * 60
                if last_ts >= end_ts:
                    break
                time.sleep(1.2)

        LOG.info(f"[KrakenClient] DONE | rows={rows_written:,} | last_ts={last_ts}")
        return rows_written, last_ts

    # ----------------------------------------------------------------------
    # MAP RAW KRAKEN CANDLES -> tuples for CSV
    # ----------------------------------------------------------------------
    def _map_chunk(self, chunk: List[List], end_ts: Optional[int] = None) -> List[tuple]:

        mapped = []
        for row in chunk:
            try:
                ts = int(row[0])
                if end_ts is not None and ts > end_ts:
                    continue
                mapped.append(
                    (
                        ts,
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[6]),  # Kraken: index 6 = volume
                    )
                )
            except Exception as e:
                LOG.error(f"[KrakenClient] Failed mapping row -> {e}")

        return mapped
