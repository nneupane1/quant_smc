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
from typing import List, Optional, Tuple
from datetime import datetime

try:  # pragma: no cover - optional pretty progress
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover
    Console = None
    Progress = None

from quant_system.utils.logger import console_stage, fmt_num, fmt_seconds, get_logger

LOG = get_logger("kraken_trades")
RICH_CONSOLE = Console() if Console is not None else None


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
        start_ts = float(start_cursor) / 1_000_000_000.0
        total_span = max(float(end_ts) - start_ts, 1.0)
        batch_no = 0
        milestone_key: Optional[str] = None
        started_at = time.perf_counter()

        progress = None
        task_id = None
        if Progress is not None and RICH_CONSOLE is not None:
            progress = Progress(
                SpinnerColumn(style="bright_cyan"),
                TextColumn("[bold bright_cyan]Kraken trades[/bold bright_cyan]"),
                BarColumn(bar_width=32),
                TaskProgressColumn(),
                TextColumn("[white]{task.fields[window]}[/white]"),
                TextColumn("[magenta]{task.fields[rows]} rows[/magenta]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=RICH_CONSOLE,
                transient=False,
            )
            progress.start()
            task_id = progress.add_task(
                "download",
                total=total_span,
                completed=0.0,
                window="-",
                rows="0",
            )

        with open(output_csv, mode, newline="") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["dt", "timestamp", "price", "volume", "side"])

            while True:
                batch_no += 1
                trades, last = self.fetch(cur)
                if not trades:
                    console_stage("Trades download stopped", "no trades returned", status="warn")
                    break

                batch_last_ts: Optional[float] = None
                for t in trades:
                    price = float(t[0])
                    volume = float(t[1])
                    ts = float(t[2])
                    batch_last_ts = ts
                    side = t[3]  # b/s
                    if ts > end_ts:
                        if progress is not None and task_id is not None:
                            progress.update(
                                task_id,
                                completed=total_span,
                                window=datetime.utcfromtimestamp(end_ts).strftime("%Y-%m"),
                                rows=fmt_num(rows_written),
                            )
                            progress.stop()
                        console_stage(
                            "Trades download complete",
                            f"rows={fmt_num(rows_written)} elapsed={fmt_seconds(time.perf_counter() - started_at)}",
                            status="ok",
                        )
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

                if batch_last_ts is not None:
                    current_month = datetime.utcfromtimestamp(batch_last_ts).strftime("%Y-%m")
                    if current_month != milestone_key:
                        milestone_key = current_month
                        console_stage(
                            "History milestone",
                            (
                                f"{current_month} | rows={fmt_num(rows_written)} "
                                f"cursor={last}"
                            ),
                            status="info",
                        )

                    if progress is not None and task_id is not None:
                        completed = max(0.0, min(batch_last_ts, float(end_ts)) - start_ts)
                        progress.update(
                            task_id,
                            completed=completed,
                            window=current_month,
                            rows=fmt_num(rows_written),
                        )
                    elif batch_no == 1 or batch_no % 25 == 0:
                        pct = (max(0.0, min(batch_last_ts, float(end_ts)) - start_ts) / total_span) * 100.0
                        console_stage(
                            "Trades progress",
                            (
                                f"{pct:5.1f}% | month={current_month} | rows={fmt_num(rows_written)} "
                                f"| elapsed={fmt_seconds(time.perf_counter() - started_at)}"
                            ),
                            status="info",
                        )

                if last == cur:
                    console_stage("Trades download stopped", "cursor did not advance", status="warn")
                    break
                cur = last
                time.sleep(sleep)

        if progress is not None and task_id is not None:
            progress.stop()
        console_stage(
            "Trades download complete",
            f"rows={fmt_num(rows_written)} elapsed={fmt_seconds(time.perf_counter() - started_at)}",
            status="ok",
        )
        return rows_written, cur
