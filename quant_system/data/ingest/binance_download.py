"""
CLI helper to download 1m OHLCV from Binance and save to CSV.

Example:
    python -m quant_system.data.ingest.binance_download \\
        --pair BTC/USDT \\
        --start 1609459200 \\
        --end 1609545600 \\
        --output data/raw_1m/BTCUSDT_1m_binance.csv
"""

import argparse
import os
import pandas as pd
from quant_system.data.ingest.binance_client import BinanceClient


def main():
    parser = argparse.ArgumentParser(description="Download Binance 1m OHLCV to CSV")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair, e.g., BTC/USDT")
    parser.add_argument("--start", type=int, required=True, help="Start timestamp (seconds UTC)")
    parser.add_argument("--end", type=int, required=True, help="End timestamp (seconds UTC)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between batches (seconds)")
    args = parser.parse_args()

    client = BinanceClient(pair=args.pair)
    rows = client.fetch_ohlcv(start_ts=args.start, end_ts=args.end, batch_sleep=args.sleep)

    if not rows:
        print("No data returned; check timestamps or pair.")
        return

    output_path = args.output or os.path.join(
        "data/raw_1m",
        f"{args.pair.replace('/', '')}_1m_binance.csv",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
