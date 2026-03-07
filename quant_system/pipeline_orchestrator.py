"""
Top-level pipeline orchestrator.

Canonical flow:
 - ingest Kraken 1m data
 - build 15m / 1h / 6h / 12h bars
 - build features
 - generate labels
 - train the full model suite
 - validate saved artifacts with a sample inference pass
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant_system.cli.common import (
    default_asset,
    default_conf_dir,
    load_registry,
    read_frame,
    resolve_conf_dir,
    save_json,
)
from quant_system.config.config_loader import ConfigLoader
from quant_system.data.ingest.ingestion import DataIngestion
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.train_orchestrator import TrainOrchestrator
from quant_system.utils.logger import (
    console_kv,
    console_rule,
    console_stage,
    get_logger,
    runtime_logged,
)

LOG = get_logger("pipeline_orchestrator")


def _parse_utc_timestamp(value: str, *, is_end: bool = False) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if is_end and len(str(value).strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int(ts.timestamp())


class PipelineOrchestrator:
    def __init__(
        self,
        conf_dir: str = "quant_system/config",
        *,
        model_registry: Optional[str] = None,
        artifact_root: str = "artifacts/pipeline/latest",
    ):
        self.conf_dir = resolve_conf_dir(conf_dir)
        self.cfg_loader = ConfigLoader(self.conf_dir)
        self.cfg = self.cfg_loader.load()
        self.registry = load_registry(self.cfg, model_registry)
        self.train_orchestrator = TrainOrchestrator(
            conf_dir=self.conf_dir,
            model_registry=model_registry,
            artifact_root=artifact_root,
        )
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.paths_cfg = dict(self.cfg.get("paths", {}) or {})
        self.assets_cfg = dict(self.cfg.get("assets", {}) or {})
        self.assets_meta = dict(
            self.assets_cfg.get("metadata")
            or self.cfg.get("metadata")
            or {}
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            import json

            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _asset_meta(self, asset: str) -> Dict[str, Any]:
        if asset not in self.assets_meta:
            raise KeyError(f"Asset {asset} is not configured in assets metadata.")
        return dict(self.assets_meta[asset] or {})

    def _default_paths(self, asset: str) -> Dict[str, Path]:
        raw_root = Path(self.paths_cfg.get("raw_1m", "data/raw_1m"))
        tf_root = Path(self.paths_cfg.get("tf", "data/tf"))
        features_root = Path(self.paths_cfg.get("features", "data/features"))
        labels_root = Path(self.paths_cfg.get("labels", "data/labels"))
        asset_artifact_root = self.artifact_root / asset
        return {
            "raw_1m": raw_root / f"{asset}_1m.csv",
            "tf_dir": tf_root,
            "features_out": features_root / f"{asset}_features.csv",
            "labels_out": labels_root / f"{asset}_labels.csv",
            "merged_out": asset_artifact_root / "training_frame.csv",
            "train_manifest_out": asset_artifact_root / "train_manifest.json",
            "pipeline_manifest_out": asset_artifact_root / "pipeline_manifest.json",
        }

    def _date_bounds(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        data_cfg = dict(self.cfg.get("data", {}) or {})
        start_value = start_date or data_cfg.get("start_date") or "2017-01-01"
        end_value = end_date or data_cfg.get("end_date") or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        return {
            "start_date": str(start_value),
            "end_date": str(end_value),
            "start_ts": _parse_utc_timestamp(str(start_value), is_end=False),
            "end_ts": _parse_utc_timestamp(str(end_value), is_end=True),
        }

    def run_asset(
        self,
        *,
        asset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        raw_out: Optional[str] = None,
        tf_dir: Optional[str] = None,
        features_out: Optional[str] = None,
        labels_out: Optional[str] = None,
        merged_out: Optional[str] = None,
        train_manifest_out: Optional[str] = None,
        pipeline_manifest_out: Optional[str] = None,
        batch_sleep: float = 1.2,
        interval: int = 1,
        resume_ingestion: bool = True,
        resume_pipeline: bool = True,
        models: Optional[list[str]] = None,
        skip_ingestion: bool = False,
        skip_training: bool = False,
        skip_validation: bool = False,
        force_pipeline_restart: bool = False,
        pipeline_checkpoint_out: Optional[str] = None,
    ) -> Dict[str, Any]:
        asset = default_asset(self.cfg, asset)
        asset_meta = self._asset_meta(asset)
        path_defaults = self._default_paths(asset)
        bounds = self._date_bounds(start_date=start_date, end_date=end_date)

        raw_out = str(raw_out or path_defaults["raw_1m"])
        tf_dir = str(tf_dir or path_defaults["tf_dir"])
        features_out = str(features_out or path_defaults["features_out"])
        labels_out = str(labels_out or path_defaults["labels_out"])
        merged_out = str(merged_out or path_defaults["merged_out"])
        train_manifest_out = str(train_manifest_out or path_defaults["train_manifest_out"])
        pipeline_manifest_out = str(pipeline_manifest_out or path_defaults["pipeline_manifest_out"])
        pipeline_checkpoint_out = str(
            pipeline_checkpoint_out
            or (Path(pipeline_manifest_out).parent / "pipeline_checkpoint.json")
        )

        manifest: Dict[str, Any] = {
            "asset": asset,
            "kraken_pair": asset_meta.get("kraken_pair"),
            "window": bounds,
            "paths": {
                "raw_1m": raw_out,
                "tf_dir": tf_dir,
                "features_out": features_out,
                "labels_out": labels_out,
                "merged_out": merged_out,
                "train_manifest_out": train_manifest_out,
                "pipeline_manifest_out": pipeline_manifest_out,
                "pipeline_checkpoint_out": pipeline_checkpoint_out,
                "model_registry": self.registry.base_dir,
            },
            "steps": {},
        }

        console_rule(f"Pipeline Room | {asset}", style="cyan")
        console_kv(
            "Pipeline Plan",
            {
                "asset": asset,
                "window": f"{bounds['start_date']} -> {bounds['end_date']}",
                "resume_ingestion": bool(resume_ingestion),
                "resume_pipeline": bool(resume_pipeline and not force_pipeline_restart),
                "raw_1m": raw_out,
                "tf_dir": tf_dir,
                "pipeline_manifest": pipeline_manifest_out,
                "pipeline_checkpoint": pipeline_checkpoint_out,
            },
            style="cyan",
        )

        checkpoint_path = Path(pipeline_checkpoint_out)

        def _persist_checkpoint(*, completed: bool = False) -> None:
            payload = {
                "asset": asset,
                "window": bounds,
                "paths": manifest["paths"],
                "manifest": manifest,
                "completed": bool(completed),
            }
            save_json(checkpoint_path, payload)

        if resume_pipeline and not force_pipeline_restart:
            checkpoint_payload = self._read_json(checkpoint_path)
            cached_paths = checkpoint_payload.get("paths", {}) if isinstance(checkpoint_payload, dict) else {}
            paths_match = (
                isinstance(cached_paths, dict)
                and cached_paths.get("raw_1m") == raw_out
                and cached_paths.get("tf_dir") == tf_dir
                and cached_paths.get("merged_out") == merged_out
                and cached_paths.get("train_manifest_out") == train_manifest_out
                and cached_paths.get("pipeline_manifest_out") == pipeline_manifest_out
            )
            if (
                checkpoint_payload.get("asset") == asset
                and checkpoint_payload.get("window") == bounds
                and paths_match
            ):
                cached_manifest = checkpoint_payload.get("manifest", {})
                if isinstance(cached_manifest, dict):
                    cached_steps = cached_manifest.get("steps", {})
                    if isinstance(cached_steps, dict):
                        manifest["steps"].update(cached_steps)
                if bool(checkpoint_payload.get("completed")) and Path(pipeline_manifest_out).exists():
                    final_manifest = self._read_json(Path(pipeline_manifest_out))
                    if isinstance(final_manifest, dict) and final_manifest.get("steps", {}).get("validation", {}).get("status") in {"completed", "skipped"}:
                        console_stage(
                            "Pipeline resume hit",
                            f"manifest={pipeline_manifest_out}",
                            status="ok",
                        )
                        return final_manifest

        cached_ingestion = manifest["steps"].get("ingestion", {})
        ingestion_reusable = (
            isinstance(cached_ingestion, dict)
            and cached_ingestion.get("status") == "completed"
            and Path(raw_out).exists()
            and all((Path(tf_dir) / f"{asset}_{tf}.csv").exists() for tf in ("15m", "1h", "6h", "12h"))
        )
        if skip_ingestion:
            manifest["steps"]["ingestion"] = {
                "status": "skipped",
                "raw_1m_path": raw_out,
                "tf_dir": tf_dir,
            }
            _persist_checkpoint()
        elif resume_pipeline and ingestion_reusable:
            manifest["steps"]["ingestion"] = {
                **cached_ingestion,
                "status": "completed",
                "resumed": True,
            }
            console_stage(
                "Ingestion checkpoint hit",
                f"raw={raw_out} tf_dir={tf_dir}",
                status="ok",
            )
            _persist_checkpoint()
        else:
            LOG.info(
                "[Pipeline] Ingestion start asset=%s kraken_pair=%s %s -> %s",
                asset,
                asset_meta.get("kraken_pair"),
                bounds["start_date"],
                bounds["end_date"],
            )
            ingestion = DataIngestion(
                pair=asset_meta.get("kraken_pair") or asset,
                start_ts=bounds["start_ts"],
                end_ts=bounds["end_ts"],
                output_path=raw_out,
                tf_output_dir=tf_dir,
                batch_sleep=batch_sleep,
                interval=interval,
                conf_dir=self.conf_dir,
                build_timeframes=True,
                resume=resume_ingestion,
            )
            manifest["steps"]["ingestion"] = {
                "status": "completed",
                **ingestion.run(),
            }
            _persist_checkpoint()

        train_manifest: Dict[str, Any] = {}
        cached_training = manifest["steps"].get("training", {})
        training_reusable = (
            isinstance(cached_training, dict)
            and cached_training.get("status") == "completed"
            and Path(train_manifest_out).exists()
            and Path(merged_out).exists()
        )
        if skip_training:
            manifest["steps"]["training"] = {
                "status": "skipped",
                "merged_out": merged_out,
            }
            _persist_checkpoint()
        elif resume_pipeline and training_reusable:
            train_manifest = self._read_json(Path(train_manifest_out))
            if train_manifest:
                manifest["steps"]["training"] = {
                    "status": "completed",
                    **train_manifest,
                    "resumed": True,
                }
            else:
                manifest["steps"]["training"] = {
                    **cached_training,
                    "status": "completed",
                    "resumed": True,
                }
            console_stage(
                "Training checkpoint hit",
                f"manifest={train_manifest_out}",
                status="ok",
            )
            _persist_checkpoint()
        else:
            LOG.info("[Pipeline] Training start asset=%s tf_dir=%s", asset, tf_dir)
            train_manifest = self.train_orchestrator.run_asset(
                asset=asset,
                tf_dir=tf_dir,
                features_out=features_out,
                labels_out=labels_out,
                merged_out=merged_out,
                manifest_out=train_manifest_out,
                models=models,
                resume=resume_pipeline,
                force_retrain=force_pipeline_restart,
                force_rebuild_frame=force_pipeline_restart,
            )
            manifest["steps"]["training"] = {
                "status": "completed",
                **train_manifest,
            }
            _persist_checkpoint()

        cached_validation = manifest["steps"].get("validation", {})
        validation_reusable = (
            isinstance(cached_validation, dict)
            and cached_validation.get("status") == "completed"
            and Path(merged_out).exists()
        )
        if skip_validation:
            manifest["steps"]["validation"] = {"status": "skipped"}
            _persist_checkpoint()
        elif resume_pipeline and validation_reusable:
            manifest["steps"]["validation"] = {
                **cached_validation,
                "status": "completed",
                "resumed": True,
            }
            console_stage(
                "Validation checkpoint hit",
                f"merged_out={merged_out}",
                status="ok",
            )
            _persist_checkpoint()
        else:
            LOG.info("[Pipeline] Validation start asset=%s", asset)
            validation = self._validate(asset=asset, merged_out=merged_out)
            manifest["steps"]["validation"] = {
                "status": "completed",
                **validation,
            }
            _persist_checkpoint()

        save_json(pipeline_manifest_out, manifest)
        _persist_checkpoint(completed=True)
        return manifest

    def _validate(self, *, asset: str, merged_out: str) -> Dict[str, Any]:
        train_df = read_frame(merged_out)
        if train_df.empty:
            raise ValueError("Validation failed: merged training frame is empty.")

        latest_row = train_df.iloc[-1].to_dict()
        predictor = ModelPredictor(self.registry)
        specialist_list = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
        predictions = predictor.predict_single(latest_row, specialist_list)

        required_models = [
            f"{asset}_liq_flow",
            f"{asset}_bos_cont",
            f"{asset}_flow_1h",
            f"{asset}_momo",
            f"{asset}_eop",
            f"{asset}_edp",
            f"{asset}_meta",
            f"{asset}_confluence",
            f"{asset}_hazard",
            f"{asset}_quantile",
        ]
        available_models = [
            model_name
            for model_name in required_models
            if (Path(self.registry.base_dir) / model_name).exists()
        ]

        sample = {
            "prob_liq_flow": predictions.get("prob_liq_flow"),
            "prob_bos_cont": predictions.get("prob_bos_cont"),
            "prob_flow_1h": predictions.get("prob_flow_1h"),
            "prob_meta": predictions.get("prob_meta"),
            "prob_confluence": predictions.get("prob_confluence"),
            "hazard_score": predictions.get("hazard_score"),
            "q05": predictions.get("q05"),
            "q50": predictions.get("q50"),
            "q95": predictions.get("q95"),
            "cvar": predictions.get("cvar"),
        }
        return {
            "rows": int(len(train_df)),
            "sample_dt": str(train_df.iloc[-1].get("dt", "")),
            "available_models": available_models,
            "sample_prediction": sample,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kraken ingestion, timeframe build, features, labels, training, and validation in one command."
    )
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Configured asset key, e.g. BTCUSD")
    parser.add_argument("--start-date", default=None, help="UTC start date/time. Falls back to config data.start_date.")
    parser.add_argument("--end-date", default=None, help="UTC end date/time. Falls back to config data.end_date.")
    parser.add_argument("--raw-out", default=None, help="Optional raw 1m CSV output override.")
    parser.add_argument("--tf-dir", default=None, help="Optional timeframe directory override.")
    parser.add_argument("--features-out", default=None, help="Optional features CSV output override.")
    parser.add_argument("--labels-out", default=None, help="Optional labels CSV output override.")
    parser.add_argument("--merged-out", default=None, help="Optional merged training frame output override.")
    parser.add_argument("--train-manifest-out", default=None, help="Optional training manifest output override.")
    parser.add_argument("--pipeline-manifest-out", default=None, help="Optional pipeline manifest output override.")
    parser.add_argument("--model-registry", default=None, help="Optional model registry override.")
    parser.add_argument("--artifact-root", default="artifacts/pipeline/latest", help="Default root for pipeline outputs.")
    parser.add_argument("--batch-sleep", default=1.2, type=float, help="Pause between Kraken requests.")
    parser.add_argument("--interval", default=1, type=int, help="Kraken OHLC interval in minutes.")
    parser.add_argument("--models", default=None, help="Optional comma-separated model list for training.")
    parser.add_argument("--no-resume-ingestion", action="store_true", help="Disable raw 1m checkpoint resume logic.")
    parser.add_argument("--no-resume-pipeline", action="store_true", help="Disable pipeline checkpoint resume logic.")
    parser.add_argument("--force-pipeline-restart", action="store_true", help="Ignore pipeline checkpoints and rerun all enabled stages.")
    parser.add_argument("--pipeline-checkpoint-out", default=None, help="Optional pipeline checkpoint output override.")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


@runtime_logged("Pipeline orchestrator runtime")
def main() -> None:
    args = parse_args()
    orchestrator = PipelineOrchestrator(
        conf_dir=args.config_dir,
        model_registry=args.model_registry,
        artifact_root=args.artifact_root,
    )
    asset = default_asset(orchestrator.cfg, args.asset)
    manifest = orchestrator.run_asset(
        asset=asset,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_out=args.raw_out,
        tf_dir=args.tf_dir,
        features_out=args.features_out,
        labels_out=args.labels_out,
        merged_out=args.merged_out,
        train_manifest_out=args.train_manifest_out,
        pipeline_manifest_out=args.pipeline_manifest_out,
        batch_sleep=args.batch_sleep,
        interval=args.interval,
        resume_ingestion=not args.no_resume_ingestion,
        resume_pipeline=not args.no_resume_pipeline,
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        skip_ingestion=args.skip_ingestion,
        skip_training=args.skip_training,
        skip_validation=args.skip_validation,
        force_pipeline_restart=args.force_pipeline_restart,
        pipeline_checkpoint_out=args.pipeline_checkpoint_out,
    )
    LOG.info(
        "[Pipeline] Complete asset=%s training=%s validation=%s manifest=%s",
        manifest["asset"],
        manifest["steps"].get("training", {}).get("status"),
        manifest["steps"].get("validation", {}).get("status"),
        manifest["paths"]["pipeline_manifest_out"],
    )


if __name__ == "__main__":
    main()
