"""
CLI helper to download historical OHLCV from Kraken using existing client/config.

Example:
    python -m quant_system.data.ingest.kraken_download ^
        --asset BTCUSD ^
        --interval 1 ^
        --start-year 2013
"""

import argparse
import datetime as dt
import os

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.kraken_client import KrakenClient
from quant_system.utils.logger import runtime_logged


def _tail_timestamp(path: str) -> int:
    """
    Read last timestamp from an existing CSV quickly.
    Returns int seconds or None.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = min(4096, end)
            f.seek(-size, os.SEEK_END)
            lines = f.read().splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                parts = line.decode().split(",")
                # if header present, skip
                if parts[0].lower() == "dt":
                    continue
                # dt,timestamp,open,... => timestamp at index 1
                ts = int(parts[1])
                return ts
    except Exception:
        return None
    return None


@runtime_logged("Kraken OHLC download runtime")
def main():
    parser = argparse.ArgumentParser(description="Download Kraken OHLCV to CSV")
    parser.add_argument("--asset", default=None, help="Asset key from assets.yaml (e.g., BTCUSD)")
    parser.add_argument("--interval", type=int, default=1, help="Interval minutes (1, 5, 15, 60, 240, 1440...)")
    parser.add_argument("--start-year", type=int, default=2013, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=None, help="End year (inclusive); defaults to current year")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append/resume from existing CSV (uses last timestamp)")
    args = parser.parse_args()

    end_year = args.end_year or dt.datetime.utcnow().year
    requested_end_ts = int(dt.datetime(end_year + 1, 1, 1).timestamp()) - args.interval * 60
    # Do not request beyond "now" (Kraken can't return future data).
    now_ts = int(dt.datetime.utcnow().timestamp())
    end_ts = min(requested_end_ts, now_ts)

    cfg = ConfigLoader(conf_dir="quant_system/config")
    client = KrakenClient(cfg)
    if args.asset:
        client.set_asset(args.asset)
    storage_cfg = cfg.load_yaml("storage.yaml").get("paths", {})
    output_path = args.output or os.path.join(
        storage_cfg.get("raw_1m", "data/raw_1m"),
        f"{client.asset}_1m.csv",
    )

    # Determine start_ts (either from CLI year or from last row if appending)
    start_ts = int(dt.datetime(args.start_year, 1, 1).timestamp())
    if args.append and os.path.exists(output_path):
        last_ts = _tail_timestamp(output_path)
        if last_ts:
            start_ts = last_ts + args.interval * 60

    client.download_range_csv(
        start_ts=start_ts,
        end_ts=end_ts,
        interval=args.interval,
        output_csv=output_path,
        append=args.append,
    )


if __name__ == "__main__":
    main()
