"""
kraken_trades.py
Simple downloader for historical trades from Kraken public API, with since-cursor paging.

Note: Kraken's OHLC endpoint for 1m only returns a small recent window. To build deep
1m history, we fetch raw trades and then resample to 1m locally.
"""

import os
import csv
import time
import requests
from typing import List, Tuple
from datetime import datetime

from quant_system.utils.logger import get_logger

LOG = get_logger("kraken_trades")


class KrakenTradesDownloader:
    def __init__(self, pair: str):
        """
        pair: Kraken altname, e.g. XBTUSD, ETHUSD, XBTUSDT
        """
        self.pair = pair
        self.base_url = "https://api.kraken.com/0/public/Trades"

    def fetch(self, since: int) -> Tuple[List[list], int]:
        """
        Fetch trades since cursor (int). Returns (trades, last_cursor).
        Kraken returns timestamps as float seconds.
        """
        params = {"pair": self.pair, "since": since}
        r = requests.get(self.base_url, params=params, timeout=30)
        j = r.json()
        if j.get("error"):
            LOG.error(f"[Trades] error: {j['error']}")
            return [], since
        result = j.get("result", {})
        # Trades are under pair key; last cursor under "last"
        pair_keys = [k for k in result.keys() if k != "last"]
        if not pair_keys:
            return [], since
        trades = result[pair_keys[0]]
        last = int(result.get("last", since))
        return trades, last

    def download_to_csv(
        self,
        output_csv: str,
        start_cursor: int,
        end_ts: float,
        sleep: float = 1.0,
        append: bool = False,
    ):
        """
        Download trades from start_cursor until trade time exceeds end_ts (float seconds).
        Writes CSV: dt, timestamp, price, volume, side.
        """
        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        mode = "a" if append and os.path.exists(output_csv) else "w"
        header_needed = not (append and os.path.exists(output_csv))

        cur = start_cursor
        rows_written = 0
        seen = set()

        with open(output_csv, mode, newline="") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["dt", "timestamp", "price", "volume", "side"])

            while True:
                trades, last = self.fetch(cur)
                if not trades:
                    LOG.info("[Trades] no trades returned, stopping.")
                    break

                for t in trades:
                    price = float(t[0])
                    volume = float(t[1])
                    ts = float(t[2])
                    side = t[3]  # b/s
                    if ts > end_ts:
                        LOG.info("[Trades] reached end_ts, stopping.")
                        return rows_written, ts
                    key = (ts, price, volume, side)
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow([
                        datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                        ts,
                        price,
                        volume,
                        side,
                    ])
                    rows_written += 1

                LOG.info(f"[Trades] wrote {rows_written} rows so far | last cursor={last}")

                if last == cur:
                    LOG.info("[Trades] cursor did not advance; stopping.")
                    break
                cur = last
                time.sleep(sleep)

        return rows_written, cur
