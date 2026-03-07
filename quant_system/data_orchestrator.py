"""
Top-level data orchestrator.

Canonical flow:
 - download Kraken 1m OHLC
 - resume from last processed timestamp when available
 - rebuild 15m / 1h / 6h / 12h bars from canonical raw 1m
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant_system.cli.common import default_asset, default_conf_dir, save_json
from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.ingestion import DataIngestion
from quant_system.utils.logger import (
    console_kv,
    console_rule,
    console_stage,
    fmt_ts,
    get_logger,
    runtime_logged,
)

LOG = get_logger("data_orchestrator")


def _parse_utc_timestamp(value: str, *, is_end: bool = False) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if is_end and len(str(value).strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int(ts.timestamp())


class DataOrchestrator:
    def __init__(self, conf_dir: str = "quant_system/config", *, artifact_root: str = "artifacts/data/latest"):
        self.conf_dir = conf_dir
        self.cfg_loader = ConfigLoader(conf_dir)
        self.cfg = self.cfg_loader.load()
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def run_asset(
        self,
        *,
        asset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        raw_out: Optional[str] = None,
        tf_dir: Optional[str] = None,
        manifest_out: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        batch_sleep: float = 1.2,
        interval: int = 1,
        resume: bool = True,
    ) -> Dict[str, Any]:
        asset = default_asset(self.cfg, asset)
        assets_meta = self.cfg.get("assets", {}).get("metadata") or self.cfg.get("metadata") or {}
        asset_meta = assets_meta.get(asset, {})
        paths_cfg = self.cfg.get("paths", {}) or {}

        start_value = start_date or self.cfg.get("data", {}).get("start_date") or "2017-01-01"
        end_value = end_date or self.cfg.get("data", {}).get("end_date") or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        raw_out = raw_out or str(Path(paths_cfg.get("raw_1m", "data/raw_1m")) / f"{asset}_1m.csv")
        tf_dir = tf_dir or str(Path(paths_cfg.get("tf", "data/tf")))
        manifest_out = manifest_out or str(self.artifact_root / asset / "data_manifest.json")

        console_rule(f"Data Room | {asset}", style="cyan")
        console_kv(
            "Data Plan",
            {
                "asset": asset,
                "kraken_pair": asset_meta.get("kraken_pair") or asset,
                "start": fmt_ts(_parse_utc_timestamp(str(start_value), is_end=False)),
                "end": fmt_ts(_parse_utc_timestamp(str(end_value), is_end=True)),
                "raw_1m": raw_out,
                "tf_dir": tf_dir,
                "resume": resume,
            },
            style="cyan",
        )

        ingestion = DataIngestion(
            pair=asset_meta.get("kraken_pair") or asset,
            start_ts=_parse_utc_timestamp(str(start_value), is_end=False),
            end_ts=_parse_utc_timestamp(str(end_value), is_end=True),
            output_path=raw_out,
            tf_output_dir=tf_dir,
            batch_sleep=batch_sleep,
            interval=interval,
            conf_dir=self.conf_dir,
            build_timeframes=True,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
        result = ingestion.run()
        manifest = {
            "asset": asset,
            "kraken_pair": asset_meta.get("kraken_pair") or asset,
            "start_date": str(start_value),
            "end_date": str(end_value),
            "interval": interval,
            **result,
        }
        save_json(manifest_out, manifest)
        console_stage(
            "Data manifest saved",
            f"manifest={manifest_out} checkpoint={manifest.get('checkpoint_path')}",
            status="ok",
        )
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Kraken 1m ingestion and TF resampling in one command.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Configured asset key, e.g. BTCUSD")
    parser.add_argument("--start-date", default=None, help="UTC start date/time. Falls back to config data.start_date.")
    parser.add_argument("--end-date", default=None, help="UTC end date/time. Falls back to config data.end_date.")
    parser.add_argument("--raw-out", default=None, help="Optional raw 1m CSV output override.")
    parser.add_argument("--tf-dir", default=None, help="Optional timeframe directory override.")
    parser.add_argument("--manifest-out", default=None, help="Optional data manifest output override.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint JSON override.")
    parser.add_argument("--artifact-root", default="artifacts/data/latest", help="Default root for data manifests.")
    parser.add_argument("--batch-sleep", default=1.2, type=float, help="Pause between Kraken requests.")
    parser.add_argument("--interval", default=1, type=int, help="Kraken OHLC interval in minutes.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume logic.")
    return parser.parse_args()


@runtime_logged("Data orchestrator runtime")
def main() -> None:
    args = parse_args()
    orchestrator = DataOrchestrator(conf_dir=args.config_dir, artifact_root=args.artifact_root)
    manifest = orchestrator.run_asset(
        asset=args.asset,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_out=args.raw_out,
        tf_dir=args.tf_dir,
        manifest_out=args.manifest_out,
        checkpoint_path=args.checkpoint_path,
        batch_sleep=args.batch_sleep,
        interval=args.interval,
        resume=not args.no_resume,
    )
    console_stage(
        "Data complete",
        f"asset={manifest['asset']} rows={manifest.get('rows')} resumed_from={manifest.get('resumed_from_ts')}",
        status="ok",
    )
    LOG.info(
        "[DataOrchestrator] Complete asset=%s resumed_from=%s checkpoint=%s",
        manifest["asset"],
        manifest.get("resumed_from_ts"),
        manifest.get("checkpoint_path"),
    )


if __name__ == "__main__":
    main()
