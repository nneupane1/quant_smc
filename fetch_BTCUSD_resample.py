"""
Fetch BTCUSD 1m data from Kraken and rebuild 15m / 1h / 6h / 12h bars.

Run:
    python fetch_BTCUSD_resample.py

What it does:
    1. tries the normal Kraken OHLC path with checkpoint resume
    2. if Kraken 1m OHLC is too shallow for the requested history,
       automatically falls back to the trades-backed bootstrap path
    3. writes canonical files under:
       - data/raw_1m/BTCUSD_1m.csv
       - data/tf/BTCUSD_15m.csv
       - data/tf/BTCUSD_1h.csv
       - data/tf/BTCUSD_6h.csv
       - data/tf/BTCUSD_12h.csv

Adjust the defaults below if needed.
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas",))

from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.builder import TimeframeBuilder
from quant_system.data.ingest.kraken_trades import KrakenTradesDownloader
from quant_system.data.ingest.kraken_trades_download import resample_1m
from quant_system.data_orchestrator import DataOrchestrator
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_ts


ASSET = "BTCUSD"
CONFIG_DIR = "quant_system/config"
BERLIN_TZ = ZoneInfo("Europe/Berlin")
START_DATE: Optional[str] = "2017-01-01"
END_DATE: Optional[str] = None


def load_cfg() -> dict:
    return ConfigLoader(CONFIG_DIR).load()


def default_tf_dir(cfg: dict) -> str:
    return str(Path((cfg.get("paths", {}) or {}).get("tf", "data/tf")))


def default_raw_dir(cfg: dict) -> str:
    return str(Path((cfg.get("paths", {}) or {}).get("raw_1m", "data/raw_1m")))


def parse_utc_timestamp(value: str, *, is_end: bool = False) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if is_end and len(str(value).strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int(ts.timestamp())


def resolve_window() -> tuple[str, str, int, int]:
    start_value = START_DATE or "2017-01-01"

    if END_DATE:
        end_value = END_DATE
    else:
        yesterday_berlin = datetime.now(BERLIN_TZ).date() - timedelta(days=1)
        end_dt_berlin = datetime.combine(yesterday_berlin, time(23, 59, 59), tzinfo=BERLIN_TZ)
        end_value = end_dt_berlin.isoformat()

    start_ts = pd.Timestamp(start_value)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(BERLIN_TZ)
    else:
        start_ts = start_ts.tz_convert(BERLIN_TZ)
    start_iso = start_ts.isoformat()

    end_ts = pd.Timestamp(end_value)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(BERLIN_TZ)
    else:
        end_ts = end_ts.tz_convert(BERLIN_TZ)
    end_iso = end_ts.isoformat()

    return start_iso, end_iso, int(start_ts.tz_convert("UTC").timestamp()), int(end_ts.tz_convert("UTC").timestamp())


def enforce_existing_window(cfg: dict, start_ts: int, end_ts: int) -> None:
    raw_dir = Path(default_raw_dir(cfg))
    raw_csv = raw_dir / f"{ASSET}_1m.csv"
    trades_csv = raw_dir / f"{ASSET}_trades.csv"
    checkpoint_path = raw_dir / f"{ASSET}_1m_checkpoint.json"

    def _trim_csv(path: Path) -> Optional[int]:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        keep = (ts >= start_ts) & (ts <= end_ts)
        trimmed = df.loc[keep].copy()
        if len(trimmed) != len(df):
            trimmed.to_csv(path, index=False)
            console_stage(
                "Trim existing data",
                f"path={path} kept_rows={len(trimmed)} removed_rows={len(df) - len(trimmed)}",
                status="warn",
            )
        if trimmed.empty:
            return None
        last_ts = pd.to_numeric(trimmed["timestamp"], errors="coerce").dropna()
        return int(float(last_ts.max())) if not last_ts.empty else None

    last_raw_ts = _trim_csv(raw_csv)
    _trim_csv(trades_csv)

    if checkpoint_path.exists():
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            cp_ts = payload.get("last_processed_ts")
            if cp_ts is not None and int(cp_ts) > end_ts:
                if last_raw_ts is not None:
                    payload["last_processed_ts"] = last_raw_ts
                    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    console_stage(
                        "Checkpoint rewound",
                        f"path={checkpoint_path} last_processed_ts={last_raw_ts}",
                        status="warn",
                    )
                else:
                    checkpoint_path.unlink(missing_ok=True)
                    console_stage(
                        "Checkpoint cleared",
                        f"path={checkpoint_path} exceeded requested end window",
                        status="warn",
                    )


def bootstrap_with_trades(cfg: dict) -> dict:
    assets_meta = (cfg.get("assets", {}) or {}).get("metadata") or {}
    asset_meta = assets_meta.get(ASSET, {})
    pair = asset_meta.get("kraken_pair") or ASSET
    start_value, end_value, start_ts, end_ts = resolve_window()

    raw_dir = Path(default_raw_dir(cfg))
    tf_dir = Path(default_tf_dir(cfg))
    raw_dir.mkdir(parents=True, exist_ok=True)
    tf_dir.mkdir(parents=True, exist_ok=True)

    trades_csv = raw_dir / f"{ASSET}_trades.csv"
    raw_csv = raw_dir / f"{ASSET}_1m.csv"

    append = trades_csv.exists()
    start_cursor = int(start_ts * 1_000_000_000)
    if append:
        try:
            prev = pd.read_csv(trades_csv, usecols=["timestamp"])
            if not prev.empty:
                start_cursor = int(float(prev["timestamp"].max()) * 1_000_000_000)
        except Exception:
            append = False

    console_stage(
        "Kraken trades bootstrap",
        f"pair={pair} start={fmt_ts(start_ts)} end={fmt_ts(end_ts)} append={append}",
        status="warn",
    )
    downloader = KrakenTradesDownloader(pair=pair)
    rows, last_ts = downloader.download_to_csv(
        output_csv=str(trades_csv),
        start_cursor=start_cursor,
        end_ts=end_ts,
        sleep=1.0,
        append=append,
    )
    resample_1m(str(trades_csv), str(raw_csv), start_ts=start_ts, end_ts=end_ts)
    console_stage("Timeframe rebuild", f"source={raw_csv} -> {tf_dir}", status="info")
    TimeframeBuilder(input_csv=str(raw_csv), output_dir=str(tf_dir), pair=ASSET).build()
    return {
        "asset": ASSET,
        "kraken_pair": pair,
        "rows": rows,
        "raw_1m": str(raw_csv),
        "trades_csv": str(trades_csv),
        "tf_dir": str(tf_dir),
        "resume_mode": "kraken_trades_bootstrap",
        "last_processed_ts": int(last_ts) if last_ts else None,
        "start_date": start_value,
        "end_date": end_value,
    }


def run() -> dict:
    cfg = load_cfg()
    start_value, end_value, start_ts, end_ts = resolve_window()
    enforce_existing_window(cfg, start_ts, end_ts)
    requested_minutes = max(1, int((end_ts - start_ts) / 60))

    console_rule("Fetch 1m BTCUSD From Kraken", style="bright_blue")
    console_kv(
        "Run Card",
        {
            "asset": ASSET,
            "config_dir": CONFIG_DIR,
            "window_berlin": f"{start_value} -> {end_value}",
            "start": fmt_ts(start_ts),
            "end": fmt_ts(end_ts),
            "requested_minutes": f"{requested_minutes:,}",
            "raw_1m": str(Path(default_raw_dir(cfg)) / f"{ASSET}_1m.csv"),
            "tf_dir": default_tf_dir(cfg),
        },
        style="bright_blue",
    )

    if requested_minutes > 2_000:
        console_stage(
            "Deep history mode",
            "wide window requested -> using Kraken trades bootstrap from 2017 instead of shallow OHLC resume",
            status="warn",
        )
        return bootstrap_with_trades(cfg)

    orchestrator = DataOrchestrator(conf_dir=CONFIG_DIR)
    try:
        manifest = orchestrator.run_asset(
            asset=ASSET,
            start_date=start_value,
            end_date=end_value,
        )
        shallow_ohlc = requested_minutes > 2_000 and int(manifest.get("rows") or 0) <= 721
        if shallow_ohlc:
            console_stage(
                "OHLC window too shallow",
                (
                    f"requested≈{requested_minutes:,} minutes but Kraken returned only "
                    f"{manifest.get('rows')} rows -> switching to trades bootstrap"
                ),
                status="warn",
            )
            return bootstrap_with_trades(cfg)
        return manifest
    except ValueError as exc:
        if "No candles returned" not in str(exc):
            raise
        console_stage(
            "OHLC history unavailable",
            "Kraken 1m OHLC is too shallow for this window, switching to trades bootstrap",
            status="warn",
        )
        return bootstrap_with_trades(cfg)


def main() -> None:
    manifest = run()
    console_stage(
        "Fetch complete",
        (
            f"asset={manifest['asset']} raw_1m={manifest.get('raw_1m', 'data/raw_1m/BTCUSD_1m.csv')} "
            f"tf_dir={manifest.get('tf_dir', 'data/tf')}"
        ),
        status="ok",
    )


if __name__ == "__main__":
    main()
