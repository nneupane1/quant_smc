"""
Kraken trades downloader with checkpointing and 1m resampling.

Usage (PowerShell example):
    python -m quant_system.data.ingest.kraken_trades_checkpoint `
      --pair XBTUSD `
      --start 2013-01-01 `
      --end 2026-01-01 `
      --out-dir data/kraken_xbtusd `
      --finalize

Behaviour:
 - Streams public Trades with the Kraken "since" cursor (ns as string).
 - Persists a checkpoint JSON so you can resume without duplication.
 - Writes per-chunk 1m aggregates to a cache CSV, then finalizes to gap-filled
   1m OHLCV and rolls up to 15m/1h/6h/12h CSVs.
 - Fills empty minutes with previous close and zero volume.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from quant_system.config.config_loader import ConfigLoader

KRAKEN_REST = "https://api.kraken.com"


# --------------------------- Utilities --------------------------- #
def to_unix(ts_str: str) -> float:
    """Parse ISO-like string to unix seconds (UTC) without local-time drift."""
    s = ts_str.strip()
    if s.endswith("Z"):
        dt = datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
    elif "T" in s and ("+" in s[10:] or "-" in s[10:]):
        dt = datetime.fromisoformat(s)
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(s + "T00:00:00").replace(tzinfo=timezone.utc) if "T" not in s else datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ns_from_unix(u: float) -> str:
    """Kraken cursor expects ns in string form."""
    return str(int(u * 1_000_000_000))


# --------------------------- Checkpoint --------------------------- #
@dataclass
class CheckpointState:
    pair: str = "XBTUSD"
    start_iso: str = ""
    end_iso: str = ""
    since_cursor: Optional[str] = None
    last_ts_iso: Optional[str] = None
    created_iso: str = ""
    updated_iso: str = ""


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.state: Optional[CheckpointState] = None

    def load(self) -> Optional[CheckpointState]:
        if not self.path.exists():
            return None
        with open(self.path, "r") as f:
            data = json.load(f)
        self.state = CheckpointState(**data)
        return self.state

    def save(self, state: CheckpointState):
        self.state = state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(asdict(state), f, indent=2)


# --------------------------- Rate limiter --------------------------- #
class RateLimiter:
    def __init__(self, min_interval_sec: float = 1.2):
        self.min_interval = min_interval_sec
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()


# --------------------------- Kraken client (public Trades) --------------------------- #
class KrakenPublic:
    def __init__(self, session: Optional[requests.Session] = None):
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "kraken-trades-checkpoint/1.0"})

    def trades(self, pair_altname: str, since_cursor: Optional[str]) -> Tuple[pd.DataFrame, str]:
        params = {"pair": pair_altname}
        if since_cursor:
            params["since"] = since_cursor
        r = self.s.get(f"{KRAKEN_REST}/0/public/Trades", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        result = j["result"]
        pair_key = next(k for k in result.keys() if k != "last")
        rows = result[pair_key]
        if not rows:
            return pd.DataFrame(columns=["price", "volume", "ts"]), result.get("last")
        df = pd.DataFrame(rows, columns=["price", "volume", "time", "side", "ordertype", "misc", "trade_id"])
        df["price"] = df["price"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["ts"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
        df = df[["price", "volume", "ts"]].sort_values("ts")
        return df, result["last"]


# --------------------------- Minute aggregator (streaming) --------------------------- #
class MinuteAggregator:
    """
    Writes per-chunk minute aggregates with first/last timestamps to a cache CSV.
    Finalization reads cache, dedupes minutes, gap-fills, and emits 1m OHLCV.
    """

    def __init__(self, cache_csv: Path):
        self.cache_csv = cache_csv
        self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_csv.exists():
            pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "first_ts", "last_ts"]).to_csv(
                self.cache_csv, index=False
            )

    @staticmethod
    def _chunk_trades_to_1m(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v", "first_ts", "last_ts"])
        t = trades.set_index("ts")
        opens = t["price"].resample("1min").first()
        highs = t["price"].resample("1min").max()
        lows = t["price"].resample("1min").min()
        closes = t["price"].resample("1min").last()
        vols = t["volume"].resample("1min").sum()
        first_ts = t.index.to_series().resample("1min").min()
        last_ts = t.index.to_series().resample("1min").max()
        out = pd.DataFrame(
            {"o": opens, "h": highs, "l": lows, "c": closes, "v": vols, "first_ts": first_ts, "last_ts": last_ts}
        ).dropna(subset=["o", "h", "l", "c"])
        out.index.name = "ts"
        return out

    def append_chunk(self, trades: pd.DataFrame) -> int:
        one_min = self._chunk_trades_to_1m(trades)
        if one_min.empty:
            return 0
        df = one_min.reset_index()
        df.to_csv(self.cache_csv, mode="a", header=False, index=False)
        return len(df)

    def finalize_unique_1m(self, start_iso: str, end_iso: str) -> pd.DataFrame:
        df = pd.read_csv(self.cache_csv, parse_dates=["ts", "first_ts", "last_ts"])
        if df.empty:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        start_dt = pd.to_datetime(start_iso, utc=True)
        end_dt = pd.to_datetime(end_iso, utc=True)
        df = df[(df["ts"] >= start_dt) & (df["ts"] <= end_dt)]
        df = df.set_index("ts")
        if df.empty:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])

        # Reduce per minute
        agg_h = df["h"].groupby(level=0).max()
        agg_l = df["l"].groupby(level=0).min()
        agg_v = df["v"].groupby(level=0).sum()
        first_idx = df["first_ts"].groupby(level=0).idxmin()
        last_idx = df["last_ts"].groupby(level=0).idxmax()
        agg_o = df.loc[first_idx, "o"].rename("o")
        agg_c = df.loc[last_idx, "c"].rename("c")
        merged = pd.concat(
            [agg_o, agg_h.rename("h"), agg_l.rename("l"), agg_c, agg_v.rename("v")],
            axis=1,
        ).sort_index()

        # Gap fill minutes
        full_index = pd.date_range(merged.index[0].floor("min"), merged.index[-1].floor("min"), freq="1min", tz="UTC")
        out = merged.reindex(full_index)
        gap_mask = out["v"].isna()
        c_ff = out["c"].ffill()
        out["c"] = c_ff
        out["o"] = out["o"].fillna(c_ff.shift(1))
        out["h"] = out["h"].fillna(c_ff)
        out["l"] = out["l"].fillna(c_ff)
        out["v"] = out["v"].fillna(0.0)
        out.index.name = "ts"
        out["o"] = out["o"].fillna(out["c"])
        out["gap_filled"] = gap_mask.astype(int)
        return out


# --------------------------- Resampler --------------------------- #
class Resampler:
    @staticmethod
    def rollup_from_1m(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        o = df_1m["o"].resample(rule, label="right", closed="right").first()
        h = df_1m["h"].resample(rule, label="right", closed="right").max()
        l = df_1m["l"].resample(rule, label="right", closed="right").min()
        c = df_1m["c"].resample(rule, label="right", closed="right").last()
        v = df_1m["v"].resample(rule, label="right", closed="right").sum()
        out = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}).dropna()
        out.index.name = "ts"
        return out


@lru_cache(maxsize=32)
def _canonical_asset_name(pair: str) -> str:
    cfg = ConfigLoader("quant_system/config").load_yaml("assets.yaml")
    for asset, meta in cfg.get("metadata", {}).items():
        if asset == pair or meta.get("kraken_pair") == pair:
            return asset
    return pair


def _finalize_ohlcv_frame(df: pd.DataFrame, last_source_ts: pd.Timestamp, include_gap: bool = False) -> pd.DataFrame:
    if df.empty:
        cols = ["dt", "timestamp", "open", "high", "low", "close", "volume"]
        if include_gap:
            cols.append("gap_filled")
        return pd.DataFrame(columns=cols)

    out = df.copy()
    if last_source_ts < out.index[-1]:
        out = out.iloc[:-1]
    if out.empty:
        cols = ["dt", "timestamp", "open", "high", "low", "close", "volume"]
        if include_gap:
            cols.append("gap_filled")
        return pd.DataFrame(columns=cols)

    out = out.reset_index()
    out["dt"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["timestamp"] = out["ts"].astype("int64") // 10**9
    out = out.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    cols = ["dt", "timestamp", "open", "high", "low", "close", "volume"]
    if include_gap and "gap_filled" in out.columns:
        cols.append("gap_filled")
    return out[cols]


# --------------------------- Orchestrator --------------------------- #
class KrakenDownloader:
    def __init__(self, pair: str, start: str, end: str, out_dir: Path, min_req_interval=1.2):
        load_dotenv()  # keys not required for public calls
        self.pair = pair
        self.start_iso = start
        self.end_iso = end
        self.start_unix = to_unix(start)
        self.end_unix = to_unix(end)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trades_total = 0
        self.minutes_total = 0
        self.started_at = time.time()

        self.chk = Checkpoint(out_dir / f"{pair}_checkpoint.json")
        st = self.chk.load()
        if st is None:
            st = CheckpointState(
                pair=pair,
                start_iso=start,
                end_iso=end,
                since_cursor=ns_from_unix(self.start_unix),
                last_ts_iso=None,
                created_iso=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                updated_iso=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            )
            self.chk.save(st)
        self.state = st

        self.api = KrakenPublic()
        self.limiter = RateLimiter(min_interval_sec=min_req_interval)
        self.agg = MinuteAggregator(out_dir / f"{pair}_1m_cache.csv")

    def fetch_loop(self, max_pages: Optional[int] = None):
        pages = 0
        while True:
            if max_pages and pages >= max_pages:
                print(f"[stop] hit max_pages={max_pages}")
                break

            self.limiter.wait()
            try:
                df, next_cursor = self.api.trades(self.pair, self.state.since_cursor)
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                wait = 5 if code == 429 else 3
                print(f"[warn] HTTP {code}, backing off {wait}s...")
                time.sleep(wait)
                continue
            except Exception as e:
                print(f"[warn] {e}; sleeping 3s...")
                time.sleep(3)
                continue

            if df.empty:
                print("[info] empty page; done for now.")
                break

            # clip to end time if exceeded
            if df["ts"].max().timestamp() >= self.end_unix:
                df = df[df["ts"].astype("int64") / 1e9 <= self.end_unix]
                mins_added = self.agg.append_chunk(df)
                self.trades_total += len(df)
                self.minutes_total += mins_added
                self.state.since_cursor = next_cursor
                self.state.last_ts_iso = df["ts"].max().isoformat()
                self.state.updated_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                self.chk.save(self.state)
                elapsed = time.time() - self.started_at
                print(
                    f"[done] reached end time at {self.state.last_ts_iso} | "
                    f"trades={self.trades_total:,} minutes={self.minutes_total:,} "
                    f"elapsed={elapsed/60:.2f}m"
                )
                break

            mins_added = self.agg.append_chunk(df)
            self.trades_total += len(df)
            self.minutes_total += mins_added
            self.state.since_cursor = next_cursor
            self.state.last_ts_iso = df["ts"].max().isoformat()
            self.state.updated_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            self.chk.save(self.state)

            pages += 1
            if pages % 5 == 0:
                elapsed = time.time() - self.started_at
                print(
                    f"[page {pages}] last_ts={self.state.last_ts_iso} | "
                    f"trades={self.trades_total:,} minutes={self.minutes_total:,} "
                    f"elapsed={elapsed/60:.2f}m"
                )

    def finalize_to_csvs(self):
        one_min = self.agg.finalize_unique_1m(self.start_iso, self.end_iso)
        if one_min.empty:
            print("[finalize] no data to write.")
            return

        asset = _canonical_asset_name(self.pair)
        last_one_min_ts = one_min.index[-1]

        # write 1m
        one_min_out = self.out_dir / f"{asset}_1m.csv"
        one_min_reset = _finalize_ohlcv_frame(one_min, last_one_min_ts, include_gap=True)
        one_min_reset.to_csv(one_min_out, index=False, float_format="%.8f")
        print(f"[write] {one_min_out} rows={len(one_min_reset):,}")

        # roll-ups
        tfs = {"15m": "15min", "1h": "1h", "6h": "6h", "12h": "12h"}
        for name, rule in tfs.items():
            bars = Resampler.rollup_from_1m(one_min, rule)
            out_path = self.out_dir / f"{asset}_{name}.csv"
            final_bars = _finalize_ohlcv_frame(bars, last_one_min_ts)
            final_bars.to_csv(out_path, index=False, float_format="%.8f")
            print(f"[write] {out_path} rows={len(final_bars):,}")


# --------------------------- CLI --------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Kraken trades -> 1m/15m/1h/6h/12h CSV (checkpointed).")
    ap.add_argument("--pair", default="XBTUSD", help="Kraken altname (default XBTUSD)")
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2013-01-01 or 2013-01-01T00:00:00")
    ap.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2026-01-01")
    ap.add_argument("--out-dir", default="data/kraken_xbtusd", help="output directory")
    ap.add_argument("--min-interval", type=float, default=1.2, help="min seconds between API calls (polite RL)")
    ap.add_argument("--max-pages", type=int, default=None, help="optional limit per run")
    ap.add_argument("--finalize", action="store_true", help="after fetch, write 1m and rolled-up CSVs")
    args = ap.parse_args()

    dl = KrakenDownloader(
        pair=args.pair,
        start=args.start,
        end=args.end,
        out_dir=Path(args.out_dir),
        min_req_interval=args.min_interval,
    )
    dl.fetch_loop(max_pages=args.max_pages)
    if args.finalize:
        dl.finalize_to_csvs()
        print("[finalize] done.")


if __name__ == "__main__":
    main()
