"""
CLI to download Kraken trades and resample to 1m OHLCV.

Example:
  python -m quant_system.data.ingest.kraken_trades_download ^
    --pair XBTUSD ^
    --start 2013-01-01 ^
    --end 2026-01-01 ^
    --trades-out data/raw_1m/XBTUSD_trades.csv ^
    --ohlcv-out data/raw_1m/XBTUSD_1m_from_trades.csv
"""

import argparse
import datetime as dt
import os
import pandas as pd

from quant_system.data.ingest.kraken_trades import KrakenTradesDownloader
from quant_system.utils.logger import runtime_logged


def parse_date(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%d").timestamp()


def resample_1m(trades_csv: str, ohlcv_out: str, start_ts: int, end_ts: int):
    df = pd.read_csv(trades_csv)
    if df.empty:
        print("No trades to resample.")
        return
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    if df.empty:
        print("No trades in requested window.")
        return
    df = df.set_index("dt").sort_index()

    # Base resample
    o = df["price"].resample("1min").first()
    h = df["price"].resample("1min").max()
    l = df["price"].resample("1min").min()
    c = df["price"].resample("1min").last()
    v = df["volume"].resample("1min").sum()

    # Build full minute index and fill gaps using previous close
    full_idx = pd.date_range(o.index.min().floor("min"), o.index.max().floor("min"), freq="1min", tz="UTC")
    o = o.reindex(full_idx)
    h = h.reindex(full_idx)
    l = l.reindex(full_idx)
    c = c.reindex(full_idx)
    v = v.reindex(full_idx).fillna(0.0)

    c_ff = c.ffill()
    o = o.fillna(c_ff.shift(1))
    h = h.fillna(c_ff)
    l = l.fillna(c_ff)
    c = c_ff

    ohlcv = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])
    ohlcv["timestamp"] = ohlcv.index.astype("int64") // 10**9
    ohlcv = ohlcv.reset_index().rename(columns={"index": "dt"})
    ohlcv["dt"] = ohlcv["dt"].dt.strftime("%Y-%m-%d %H:%M")
    ohlcv = ohlcv[["dt", "timestamp", "open", "high", "low", "close", "volume"]]
    ohlcv.to_csv(ohlcv_out, index=False)
    print(f"Resampled 1m OHLCV written to {ohlcv_out} ({len(ohlcv)} rows)")


@runtime_logged("Kraken trades download runtime")
def main():
    ap = argparse.ArgumentParser(description="Download Kraken trades and resample to 1m")
    ap.add_argument("--pair", default="XBTUSD", help="Kraken altname, e.g. XBTUSD, XBTUSDT")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--trades-out", default=None, help="Trades CSV output")
    ap.add_argument("--ohlcv-out", default=None, help="Resampled OHLCV output")
    ap.add_argument("--append", action="store_true", help="Append to existing trades CSV")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls")
    args = ap.parse_args()

    start_ts = int(parse_date(args.start))
    end_ts = int(parse_date(args.end))

    # start cursor: if appending and file exists, pick up from last timestamp; else use start_ts
    start_cursor = int(start_ts * 1e9)
    if args.append and os.path.exists(args.trades_out):
        try:
            df_prev = pd.read_csv(args.trades_out)
            if not df_prev.empty:
                last_ts = df_prev["timestamp"].max()
                start_cursor = int(last_ts * 1e9)
        except Exception:
            pass

    pair_label = args.pair.replace("/", "")
    trades_out = args.trades_out or f"data/raw_1m/{pair_label}_trades.csv"
    ohlcv_out = args.ohlcv_out or f"data/raw_1m/{pair_label}_1m_from_trades.csv"
    os.makedirs(os.path.dirname(trades_out), exist_ok=True)

    dl = KrakenTradesDownloader(pair=args.pair)
    rows, last_ts = dl.download_to_csv(
        output_csv=trades_out,
        start_cursor=start_cursor,
        end_ts=end_ts,
        sleep=args.sleep,
        append=args.append,
    )
    print(f"Trades downloaded: {rows}, last_ts={last_ts}")
    resample_1m(trades_out, ohlcv_out, start_ts=start_ts, end_ts=end_ts)


if __name__ == "__main__":
    main()
