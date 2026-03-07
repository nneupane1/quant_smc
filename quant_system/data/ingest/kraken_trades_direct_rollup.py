#!/usr/bin/env python3
"""
Kraken XBTUSD trades -> DIRECT rollup to 15m/1h/6h/12h CSVs (no 1m persisted),
with checkpoint + resume, pretty console logs, and a single-line progress bar.

Usage (PowerShell example):
  python -m quant_system.data.ingest.kraken_trades_direct_rollup `
    --pair XBTUSD `
    --start 2017-01-01 `
    --end 2026-01-01 `
    --out-dir data/kraken_xbtusd_direct
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import runtime_logged

# Silence pandas future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Pretty console (fallback to plain)
try:
    from rich.console import Console
    from rich.theme import Theme
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, SpinnerColumn, TextColumn

    console = Console(theme=Theme({"dim": "dim", "ok": "green", "warn": "yellow", "err": "bold red"}))
except ImportError:
    class _PlainConsole:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, *args, **kwargs):
            print("-" * 101)

    Console = _PlainConsole  # type: ignore
    Progress = None
    console = Console()  # type: ignore

KRAKEN_REST = "https://api.kraken.com"


# --------------------------- Utilities --------------------------- #
@lru_cache(maxsize=32)
def canonical_asset_name(pair: str) -> str:
    cfg = ConfigLoader("quant_system/config").load_yaml("assets.yaml")
    for asset, meta in cfg.get("metadata", {}).items():
        if asset == pair or meta.get("kraken_pair") == pair:
            return asset
    return pair


def to_unix(ts_str: str) -> float:
    """
    Parse date/time string as UTC by default to avoid local-time drift.
    Accepts: 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS', '...Z', or ISO with offset.
    """
    s = ts_str.strip()
    if s.endswith("Z"):
        dt = datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
    elif "T" in s and ("+" in s[10:] or "-" in s[10:]):  # has timezone offset
        dt = datetime.fromisoformat(s)
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(s + "T00:00:00").replace(tzinfo=timezone.utc) if "T" not in s else datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ns_from_unix(u: float) -> str:
    return str(int(u * 1_000_000_000))


def short_ts(ts: Optional[str]) -> str:
    if not ts:
        return "--"
    try:
        return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return str(ts)


# --------------------------- Console helpers --------------------------- #
def banner(pair: str, start: str, end: str, out_dir: Path, min_interval: float):
    console.rule()
    console.print(f"[bold]KRKN ▸ {pair} Direct Rollup[/bold]   |   Window: {start} → {end} (UTC)")
    console.print(f"Out: {out_dir} | Rate: ≥{min_interval:.1f}s/call   | Mode: 15m/1h/6h/12h")
    console.rule()


def log_boot(checkpoint_path: Path, cursor: Optional[str], last_minute: Optional[str], last_close: Optional[float]):
    console.print(f"[dim]{time.strftime('[%H:%M:%S]')}[/dim] boot     ▸ checkpoint: {checkpoint_path}")
    console.print(
        f"[dim]{time.strftime('[%H:%M:%S]')}[/dim] boot     ▸ resume from cursor={cursor} "
        f"last_minute={short_ts(last_minute)}  last_close={last_close}"
    )


def log_done(trades_total: int, totals: Dict[str, int], out_dir: Path, pair: str, end_ts: str):
    end_ts_z = pd.to_datetime(end_ts, utc=True).strftime("%Y-%m-%d %H:%M:%SZ")
    console.print(f"[dim]{time.strftime('[%H:%M:%S]')}[/dim] [ok]done[/ok]     ▸ reached end window {end_ts_z}")
    console.print(
        f"           totals   ▸ trades: {trades_total:,}\n"
        f"                      15m bars: {totals.get('15m',0):,} | 1h bars: {totals.get('1h',0):,} | "
        f"6h bars: {totals.get('6h',0):,} | 12h bars: {totals.get('12h',0):,}"
    )
    console.print(
        f"           files    ▸ {out_dir}/{pair}_15m.csv, {out_dir}/{pair}_1h.csv, "
        f"{out_dir}/{pair}_6h.csv, {out_dir}/{pair}_12h.csv"
    )
    console.rule()


# --------------------------- Checkpoint --------------------------- #
@dataclass
class CheckpointState:
    pair: str = "XBTUSD"
    start_iso: str = ""
    end_iso: str = ""
    since_cursor: Optional[str] = None
    last_trade_ts_iso: Optional[str] = None
    last_minute_ts_iso: Optional[str] = None
    last_close: Optional[float] = None
    # buffered partial bars (include dv)
    buf_15m_ts_iso: Optional[str] = None
    buf_15m_open: Optional[float] = None
    buf_15m_high: Optional[float] = None
    buf_15m_low: Optional[float] = None
    buf_15m_close: Optional[float] = None
    buf_15m_vol: Optional[float] = None
    buf_15m_dv: Optional[float] = None
    buf_1h_ts_iso: Optional[str] = None
    buf_1h_open: Optional[float] = None
    buf_1h_high: Optional[float] = None
    buf_1h_low: Optional[float] = None
    buf_1h_close: Optional[float] = None
    buf_1h_vol: Optional[float] = None
    buf_1h_dv: Optional[float] = None
    buf_6h_ts_iso: Optional[str] = None
    buf_6h_open: Optional[float] = None
    buf_6h_high: Optional[float] = None
    buf_6h_low: Optional[float] = None
    buf_6h_close: Optional[float] = None
    buf_6h_vol: Optional[float] = None
    buf_6h_dv: Optional[float] = None
    buf_12h_ts_iso: Optional[str] = None
    buf_12h_open: Optional[float] = None
    buf_12h_high: Optional[float] = None
    buf_12h_low: Optional[float] = None
    buf_12h_close: Optional[float] = None
    buf_12h_vol: Optional[float] = None
    buf_12h_dv: Optional[float] = None
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
        self.s.headers.update({"User-Agent": "kraken-trades-direct-rollup/1.0"})

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


# --------------------------- Direct Rollup Writer --------------------------- #
class DirectRollupWriter:
    """
    Streams pages of trades:
      1) bucket to 1-minute with gap-fill (continuity from last_close)
      2) roll up to 15m/1h/6h/12h (sum DollarVolume)
      3) write completed TF bars; buffer last partial bar per TF
    """

    def __init__(self, out_dir: Path, cp: Checkpoint):
        self.out_dir = out_dir
        self.cp = cp
        self.row_totals: Dict[str, int] = {}

        # Ensure CSV headers exist (DollarVolume included)
        for name in ["15m", "1h", "6h", "12h"]:
            p = self.out_dir / f"{self._asset_name()}_{name}.csv"
            if not p.exists():
                pd.DataFrame(
                    columns=["dt", "timestamp", "open", "high", "low", "close", "volume", "dollar_volume"]
                ).to_csv(p, index=False)
                self.row_totals[name] = 0
            else:
                with p.open("r", encoding="utf-8") as f:
                    self.row_totals[name] = max(sum(1 for _ in f) - 1, 0)

    def _append_csv(self, name: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        out = self.out_dir / f"{self._asset_name()}_{name}.csv"
        df.to_csv(out, mode="a", header=False, index=False, float_format="%.8f")
        added = len(df)
        self.row_totals[name] += added
        return added

    def _asset_name(self) -> str:
        return canonical_asset_name(self.cp.state.pair)

    def _load_buffer(self, tf: str):
        s = self.cp.state
        key = f"buf_{tf}_"
        ts = getattr(s, key + "ts_iso")
        if ts is None:
            return None
        return {
            "ts": pd.Timestamp(ts, tz="UTC"),
            "o": getattr(s, key + "open"),
            "h": getattr(s, key + "high"),
            "l": getattr(s, key + "low"),
            "c": getattr(s, key + "close"),
            "v": getattr(s, key + "vol"),
            "dv": getattr(s, key + "dv"),
        }

    def _save_buffer(self, tf: str, bar: Optional[Dict]):
        s = self.cp.state
        key = f"buf_{tf}_"
        if bar is None:
            for suffix in ["ts_iso", "open", "high", "low", "close", "vol", "dv"]:
                setattr(s, key + suffix, None)
        else:
            setattr(s, key + "ts_iso", bar["ts"].isoformat())
            setattr(s, key + "open", float(bar["o"]))
            setattr(s, key + "high", float(bar["h"]))
            setattr(s, key + "low", float(bar["l"]))
            setattr(s, key + "close", float(bar["c"]))
            setattr(s, key + "vol", float(bar["v"]))
            setattr(s, key + "dv", float(bar["dv"]))

    @staticmethod
    def _chunk_to_1min(trades: pd.DataFrame, last_minute: Optional[pd.Timestamp], last_close: Optional[float]):
        if trades.empty:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"]), last_minute, last_close

        t = trades.set_index("ts")
        opens = t["price"].resample("1min").first()
        highs = t["price"].resample("1min").max()
        lows = t["price"].resample("1min").min()
        closes = t["price"].resample("1min").last()
        vols = t["volume"].resample("1min").sum()
        one = pd.DataFrame({"o": opens, "h": highs, "l": lows, "c": closes, "v": vols})

        first_min = one.index.min().floor("min")
        last_min = one.index.max().floor("min")
        if last_minute is not None and last_close is not None:
            start_idx = min(first_min, last_minute + pd.Timedelta(minutes=1))
            prev_close = last_close
        else:
            start_idx = first_min
            prev_close = None

        full_index = pd.date_range(start_idx, last_min, freq="1min", tz="UTC")
        one = one.reindex(full_index)

        c_ff = one["c"].ffill()
        if prev_close is not None:
            c_ff = c_ff.fillna(prev_close)
        one["c"] = c_ff
        one["o"] = one["o"].fillna(one["c"].shift(1).fillna(one["c"]))
        one["h"] = one["h"].fillna(one["c"])
        one["l"] = one["l"].fillna(one["c"])
        one["v"] = one["v"].fillna(0.0)
        one.index.name = "ts"

        new_last_minute = last_min
        new_last_close = float(one["c"].iloc[-1])
        return one, new_last_minute, new_last_close

    @staticmethod
    def _rollup(one_min: pd.DataFrame, rule: str) -> pd.DataFrame:
        o = one_min["o"].resample(rule, label="right", closed="right").first()
        h = one_min["h"].resample(rule, label="right", closed="right").max()
        l = one_min["l"].resample(rule, label="right", closed="right").min()
        c = one_min["c"].resample(rule, label="right", closed="right").last()
        v = one_min["v"].resample(rule, label="right", closed="right").sum()
        dv = (one_min["c"] * one_min["v"]).resample(rule, label="right", closed="right").sum()
        out = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v, "dv": dv}).dropna()
        out.index.name = "ts"
        return out

    def _merge_buffer(self, tf: str, bars: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict]]:
        buf = self._load_buffer(tf)
        if buf is not None and not bars.empty and bars.index[0] == buf["ts"]:
            first = bars.iloc[0]
            merged = {
                "ts": buf["ts"],
                "o": buf["o"],
                "h": max(buf["h"], float(first["h"])),
                "l": min(buf["l"], float(first["l"])),
                "c": float(first["c"]),
                "v": float(buf["v"]) + float(first["v"]),
                "dv": float((buf.get("dv") or 0.0)) + float(first.get("dv", first["c"] * first["v"])),
            }
            bars = bars.iloc[1:]
            bars = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "o": [merged["o"]],
                            "h": [merged["h"]],
                            "l": [merged["l"]],
                            "c": [merged["c"]],
                            "v": [merged["v"]],
                            "dv": [merged["dv"]],
                        },
                        index=pd.DatetimeIndex([merged["ts"]], tz="UTC", name="ts"),
                    ),
                    bars,
                ]
            )

        if bars.empty:
            return pd.DataFrame(columns=["dt", "timestamp", "open", "high", "low", "close", "volume", "dollar_volume"]), buf

        partial_ts = bars.index[-1]
        partial = {
            "ts": partial_ts,
            "o": float(bars.iloc[-1]["o"]),
            "h": float(bars.iloc[-1]["h"]),
            "l": float(bars.iloc[-1]["l"]),
            "c": float(bars.iloc[-1]["c"]),
            "v": float(bars.iloc[-1]["v"]),
            "dv": float(bars.iloc[-1]["dv"]),
        }
        to_write = bars.iloc[:-1].copy()

        if not to_write.empty:
            out = to_write.reset_index()
            out["dt"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
            out["timestamp"] = out["ts"].astype("int64") // 10**9
            out = out.rename(
                columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "dv": "dollar_volume"}
            )
            out = out[["dt", "timestamp", "open", "high", "low", "close", "volume", "dollar_volume"]]
        else:
            out = pd.DataFrame(columns=["dt", "timestamp", "open", "high", "low", "close", "volume", "dollar_volume"])

        return out, partial

    def write_chunk(self, trades_chunk: pd.DataFrame, end_iso: str) -> Dict[str, int]:
        last_minute = pd.Timestamp(self.cp.state.last_minute_ts_iso, tz="UTC") if self.cp.state.last_minute_ts_iso else None
        one, new_last_minute, new_last_close = self._chunk_to_1min(trades_chunk, last_minute, self.cp.state.last_close)
        if one.empty:
            return {"15m": 0, "1h": 0, "6h": 0, "12h": 0}
        end_ts = pd.to_datetime(end_iso, utc=True)
        one = one[one.index < end_ts]
        if one.empty:
            return {"15m": 0, "1h": 0, "6h": 0, "12h": 0}

        rolled = {
            "15m": self._rollup(one, "15min"),
            "1h": self._rollup(one, "1h"),
            "6h": self._rollup(one, "6h"),
            "12h": self._rollup(one, "12h"),
        }

        bars_added: Dict[str, int] = {}
        for name in ["15m", "1h", "6h", "12h"]:
            to_write, partial = self._merge_buffer(name, rolled[name])
            bars_added[name] = self._append_csv(name, to_write)
            self._save_buffer(name, partial)

        self.cp.state.last_minute_ts_iso = new_last_minute.isoformat()
        self.cp.state.last_close = new_last_close
        return bars_added


# --------------------------- Orchestrator --------------------------- #
class KrakenDirectDownloader:
    def __init__(self, pair: str, start: str, end: str, out_dir: Path, min_req_interval=1.2):
        load_dotenv()
        self.pair = pair
        self.start_iso = start
        self.end_iso = end
        self.start_unix = to_unix(start)
        self.end_unix = to_unix(end)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trades_total = 0
        self.started_at = time.time()
        self.finished = False

        self.chk = Checkpoint(out_dir / f"{pair}_checkpoint_direct.json")
        st = self.chk.load()
        if st is None:
            st = CheckpointState(
                pair=pair,
                start_iso=start,
                end_iso=end,
                since_cursor=ns_from_unix(self.start_unix),
                last_trade_ts_iso=None,
                last_minute_ts_iso=None,
                last_close=None,
                created_iso=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                updated_iso=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            )
            self.chk.save(st)
        self.state = st

        self.api = KrakenPublic()
        self.limiter = RateLimiter(min_interval_sec=min_req_interval)
        self.writer = DirectRollupWriter(out_dir, self.chk)

    def fetch_loop(self, max_pages: Optional[int] = None):
        """
        Progress-bar driven loop (single-line UI). Still checkpoints every page.
        """
        pages = 0
        start_dt = pd.Timestamp(self.start_iso, tz="UTC")
        end_dt = pd.Timestamp(self.end_iso, tz="UTC")
        total_span = max((end_dt - start_dt).total_seconds(), 1.0)

        # Build a live progress bar if rich is available
        if Progress is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]KRKN[/]"),
                BarColumn(bar_width=None),
                TextColumn("{task.percentage:>5.1f}%"),
                TextColumn(" • trades:{task.fields[trades]:,}"),
                TextColumn(
                    " • 15m:{task.fields[b15]:,} 1h:{task.fields[b1h]:,} 6h:{task.fields[b6h]:,} 12h:{task.fields[b12h]:,}"
                ),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
        else:
            progress = None

        # Fallback context manager
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, *a): return False
            def add_task(self, *a, **k): return 0
            def update(self, *a, **k): pass

        ctx = progress if progress is not None else _Noop()

        with ctx:
            if progress is not None:
                task = progress.add_task(
                    f"{self.pair} {start_dt.strftime('%Y-%m-%d')}→{end_dt.strftime('%Y-%m-%d')}",
                    total=total_span,
                    trades=self.trades_total,
                    b15=self.writer.row_totals.get("15m", 0),
                    b1h=self.writer.row_totals.get("1h", 0),
                    b6h=self.writer.row_totals.get("6h", 0),
                    b12h=self.writer.row_totals.get("12h", 0),
                )
            else:
                task = None

            while True:
                if max_pages and pages >= max_pages:
                    console.print(f"[warn]{time.strftime('[%H:%M:%S]')} stop ▸ hit max_pages={max_pages}[/warn]")
                    break

                self.limiter.wait()
                try:
                    df, next_cursor = self.api.trades(self.pair, self.state.since_cursor)
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else None
                    console.print(f"[warn]{time.strftime('[%H:%M:%S]')} HTTP {code} ▸ backing off…[/warn]")
                    time.sleep(5 if code == 429 else 3)
                    continue
                except Exception as e:
                    console.print(f"[warn]{time.strftime('[%H:%M:%S]')} transient error: {e} ▸ sleep 3s[/warn]")
                    time.sleep(3)
                    continue

                if df.empty:
                    break

                # Clip to end window
                if df["ts"].max() >= end_dt:
                    df = df[df["ts"] < end_dt]

                if not df.empty:
                    _ = self.writer.write_chunk(df, self.end_iso)
                    self.trades_total += len(df)
                    self.state.last_trade_ts_iso = df["ts"].max().isoformat()

                    if progress is not None and task is not None:
                        covered_sec = float(
                            (pd.Timestamp(self.state.last_trade_ts_iso, tz="UTC") - start_dt).total_seconds()
                        )
                        progress.update(
                            task,
                            completed=min(max(covered_sec, 0.0), total_span),
                            trades=self.trades_total,
                            b15=self.writer.row_totals.get("15m", 0),
                            b1h=self.writer.row_totals.get("1h", 0),
                            b6h=self.writer.row_totals.get("6h", 0),
                            b12h=self.writer.row_totals.get("12h", 0),
                            refresh=True,
                        )

                # Persist checkpoint
                self.state.since_cursor = next_cursor
                self.state.updated_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                self.chk.save(self.state)

                pages += 1

                # End condition
                if not df.empty and df["ts"].max() >= end_dt - pd.Timedelta(seconds=1):
                    self.finished = True
                    break

        # One compact checkpoint line after the bar clears
        console.print(
            f"[dim]{time.strftime('[%H:%M:%S]')}[/dim] chkpt    ▸ saved (ts={short_ts(self.state.last_trade_ts_iso)})"
        )


# --------------------------- CLI --------------------------- #
@runtime_logged("Kraken direct rollup runtime")
def main():
    ap = argparse.ArgumentParser(description="Kraken trades -> DIRECT 15m/1h/6h/12h CSVs (checkpointed, no 1m persisted).")
    ap.add_argument("--pair", default="XBTUSD", help="Kraken altname (default XBTUSD)")
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2017-01-01 or 2017-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2026-01-01")
    ap.add_argument("--out-dir", default="data/kraken_xbtusd_direct", help="output directory")
    ap.add_argument("--min-interval", type=float, default=1.2, help="min seconds between API calls")
    ap.add_argument("--max-pages", type=int, default=None, help="optional page cap per run")
    args = ap.parse_args()

    dl = KrakenDirectDownloader(
        pair=args.pair,
        start=args.start,
        end=args.end,
        out_dir=Path(args.out_dir),
        min_req_interval=args.min_interval,
    )

    banner(args.pair, args.start, args.end, Path(args.out_dir), args.min_interval)
    log_boot(Path(args.out_dir) / f"{args.pair}_checkpoint_direct.json", dl.state.since_cursor, dl.state.last_minute_ts_iso, dl.state.last_close)

    try:
        dl.fetch_loop(max_pages=args.max_pages)
    except KeyboardInterrupt:
        console.print("\n[warn]Interrupted — checkpoint saved. Resume anytime.[/warn]")

    log_done(dl.trades_total, dl.writer.row_totals, Path(args.out_dir), args.pair, args.end)


if __name__ == "__main__":
    main()
