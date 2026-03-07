"""
Top-level training orchestrator.

Canonical flow:
 - build or load 15m feature frame from timeframe CSVs
 - generate or load labels
 - merge into one training frame
 - train the configured specialist/meta/confluence/hazard/quantile suite
 - persist a training manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from quant_system.cli.common import (
    default_asset,
    default_conf_dir,
    load_or_build_features,
    load_or_build_labels,
    load_registry,
    read_frame,
    resolve_conf_dir,
    save_json,
)
from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.training.model_trainer import ModelTrainer
from quant_system.utils.logger import (
    console_kv,
    console_rule,
    console_stage,
    fmt_num,
    get_logger,
    runtime_logged,
)

LOG = get_logger("train_orchestrator")


def _merge_training_frame(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df is features_df:
        return labels_df.copy()
    if "dt" in features_df.columns and "dt" in labels_df.columns:
        drop_cols = [c for c in labels_df.columns if c in features_df.columns and c != "dt"]
        merged = features_df.merge(labels_df.drop(columns=drop_cols), on="dt", how="inner")
    else:
        merged = labels_df.copy()
    return merged.reset_index(drop=True)


class TrainOrchestrator:
    def __init__(
        self,
        conf_dir: str = "quant_system/config",
        *,
        model_registry: Optional[str] = None,
        artifact_root: str = "artifacts/train/latest",
    ):
        self.conf_dir = resolve_conf_dir(conf_dir)
        self.cfg_loader = ConfigLoader(self.conf_dir)
        self.cfg = self.cfg_loader.load()
        self.registry = load_registry(self.cfg, model_registry)
        self.trainer = ModelTrainer(self.cfg_loader, self.registry)
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except Exception:
            return None

    @classmethod
    def _is_up_to_date(cls, target: Path, sources: List[Path]) -> bool:
        if not target.exists():
            return False
        t_mtime = cls._mtime(target)
        if t_mtime is None:
            return False
        for src in sources:
            s_mtime = cls._mtime(src)
            if s_mtime is not None and s_mtime > t_mtime:
                return False
        return True

    @staticmethod
    def _requested_registry_models(asset: str, requested_models: List[str]) -> List[str]:
        names: List[str] = []
        for model in requested_models:
            if model == "meta_model":
                names.append(f"{asset}_meta")
            elif model == "confluence_model":
                names.append(f"{asset}_confluence")
            else:
                names.append(f"{asset}_{model}")
        return names

    def build_training_frame(
        self,
        *,
        asset: str,
        tf_dir: Optional[str] = None,
        features_csv: Optional[str] = None,
        labels_csv: Optional[str] = None,
        features_out: Optional[str] = None,
        labels_out: Optional[str] = None,
        merged_out: Optional[str] = None,
        resume: bool = True,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        merged_path = Path(merged_out) if merged_out else None
        if (
            resume
            and not force_rebuild
            and merged_path is not None
            and merged_path.exists()
        ):
            source_paths: List[Path] = []
            if features_csv:
                source_paths.append(Path(features_csv))
            elif features_out:
                source_paths.append(Path(features_out))
            elif tf_dir:
                source_paths.extend(
                    [
                        Path(tf_dir) / f"{asset}_{tf}.csv"
                        for tf in ("15m", "1h", "6h", "12h")
                    ]
                )
            if labels_csv:
                source_paths.append(Path(labels_csv))
            elif labels_out:
                source_paths.append(Path(labels_out))

            if self._is_up_to_date(merged_path, source_paths):
                console_stage(
                    "Training frame cache hit",
                    f"path={merged_path}",
                    status="ok",
                )
                return read_frame(str(merged_path))

        effective_features_csv = features_csv
        if (
            resume
            and not force_rebuild
            and effective_features_csv is None
            and features_out
            and Path(features_out).exists()
        ):
            effective_features_csv = features_out

        features_df = load_or_build_features(
            self.cfg_loader,
            asset=asset,
            features_csv=effective_features_csv,
            tf_dir=tf_dir,
            features_out=features_out,
        )

        effective_labels_csv = labels_csv
        if (
            resume
            and not force_rebuild
            and effective_labels_csv is None
            and labels_out
            and Path(labels_out).exists()
        ):
            features_stamp_path = Path(effective_features_csv) if effective_features_csv else (Path(features_out) if features_out else None)
            labels_path = Path(labels_out)
            if features_stamp_path is None or self._is_up_to_date(labels_path, [features_stamp_path]):
                effective_labels_csv = labels_out

        labels_df = load_or_build_labels(
            self.cfg_loader,
            features_df=features_df,
            labels_csv=effective_labels_csv,
            labels_out=labels_out,
        )
        train_df = _merge_training_frame(features_df, labels_df)

        if merged_out:
            out_path = Path(merged_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(out_path, index=False)
        return train_df

    def run_asset(
        self,
        *,
        asset: str,
        tf_dir: Optional[str] = None,
        features_csv: Optional[str] = None,
        labels_csv: Optional[str] = None,
        features_out: Optional[str] = None,
        labels_out: Optional[str] = None,
        merged_out: Optional[str] = None,
        manifest_out: Optional[str] = None,
        model_state_out: Optional[str] = None,
        models: Optional[List[str]] = None,
        resume: bool = True,
        force_retrain: bool = False,
        force_rebuild_frame: bool = False,
    ) -> Dict[str, Any]:
        merged_out = merged_out or str(self.artifact_root / asset / "training_frame.csv")
        manifest_out = manifest_out or str(self.artifact_root / asset / "train_manifest.json")
        model_state_out = model_state_out or str(self.artifact_root / asset / "model_state.json")

        requested_models = ModelTrainer._normalize_requested_models(models)
        merged_path = Path(merged_out)
        manifest_path = Path(manifest_out)
        model_state_path = Path(model_state_out)

        frame_sources: List[Path] = []
        if features_csv:
            frame_sources.append(Path(features_csv))
        elif features_out:
            frame_sources.append(Path(features_out))
        elif tf_dir:
            frame_sources.extend([Path(tf_dir) / f"{asset}_{tf}.csv" for tf in ("15m", "1h", "6h", "12h")])
        if labels_csv:
            frame_sources.append(Path(labels_csv))
        elif labels_out:
            frame_sources.append(Path(labels_out))

        if resume and not force_retrain:
            cached_manifest = self._read_json(manifest_path)
            if (
                cached_manifest.get("asset") == asset
                and merged_path.exists()
                and model_state_path.exists()
                and self._is_up_to_date(merged_path, frame_sources)
                and self._is_up_to_date(manifest_path, [merged_path])
            ):
                cached_models = set(cached_manifest.get("trained_models", []) or []) | set(
                    cached_manifest.get("dependency_models", []) or []
                )
                requested_set = set(requested_models)
                version = str(cached_manifest.get("version") or "")
                registry_names = self._requested_registry_models(asset, requested_models)
                registry_ok = bool(version) and all(
                    (Path(self.registry.base_dir) / name / version).exists() for name in registry_names
                )
                if requested_set.issubset(cached_models) and registry_ok:
                    console_stage(
                        "Training resume hit",
                        f"version={version} manifest={manifest_path}",
                        status="ok",
                    )
                    return cached_manifest

        console_rule(f"Training Room | {asset}", style="green")
        console_kv(
            "Training Plan",
            {
                "asset": asset,
                "tf_dir": tf_dir or "-",
                "features_csv": features_csv or "-",
                "labels_csv": labels_csv or "-",
                "requested_models": ", ".join(requested_models),
                "registry": self.registry.base_dir,
                "resume": bool(resume and not force_retrain),
            },
            style="green",
        )
        LOG.info("[TrainOrchestrator] Building training frame for asset=%s", asset)
        console_stage("Build training frame", "features -> labels -> merged frame", status="info")
        train_df = self.build_training_frame(
            asset=asset,
            tf_dir=tf_dir,
            features_csv=features_csv,
            labels_csv=labels_csv,
            features_out=features_out,
            labels_out=labels_out,
            merged_out=merged_out,
            resume=resume,
            force_rebuild=force_rebuild_frame,
        )
        console_stage(
            "Training frame ready",
            f"rows={fmt_num(len(train_df))} merged_out={merged_out}",
            status="ok",
        )

        LOG.info("[TrainOrchestrator] Training model suite for asset=%s rows=%s", asset, len(train_df))
        console_stage("Model training", f"requested={', '.join(requested_models)}", status="info")
        train_result = self.trainer.train_asset_bundle(train_df, asset, selected_models=models)
        version = str(train_result["version"])
        metrics = self._collect_metrics(asset, version)
        model_state = {
            "asset": asset,
            "version": version,
            "requested_models": train_result.get("requested_models", []),
            "trained_models": train_result.get("trained_models", []),
            "dependency_models": train_result.get("dependency_models", []),
            "registry_dir": self.registry.base_dir,
        }
        save_json(model_state_out, model_state)
        console_stage(
            "Model state saved",
            f"version={version} state={model_state_out}",
            status="ok",
        )

        manifest = {
            "asset": asset,
            "version": version,
            "rows": int(len(train_df)),
            "registry_dir": self.registry.base_dir,
            "tf_dir": tf_dir,
            "features_csv": features_csv,
            "labels_csv": labels_csv,
            "features_out": features_out,
            "labels_out": labels_out,
            "merged_out": str(merged_out),
            "model_state_out": str(model_state_out),
            "requested_models": train_result.get("requested_models", []),
            "trained_models": train_result.get("trained_models", []),
            "dependency_models": train_result.get("dependency_models", []),
            "metrics": metrics,
            "governance": self._governance_status(metrics),
        }
        save_json(manifest_out, manifest)
        console_kv(
            "Training Summary",
            {
                "version": version,
                "trained_models": ", ".join(manifest["trained_models"]) or "-",
                "dependency_models": ", ".join(manifest["dependency_models"]) or "-",
                "manifest": manifest_out,
            },
            style="green",
        )
        return manifest

    def run(self, asset_frames: Mapping[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for asset, payload in asset_frames.items():
            if isinstance(payload, pd.DataFrame):
                merged_out = self.artifact_root / asset / "training_frame.csv"
                merged_out.parent.mkdir(parents=True, exist_ok=True)
                payload.to_csv(merged_out, index=False)
                train_result = self.trainer.train_asset_bundle(payload, asset)
                version = str(train_result["version"])
                metrics = self._collect_metrics(asset, version)
                model_state_out = self.artifact_root / asset / "model_state.json"
                save_json(
                    model_state_out,
                    {
                        "asset": asset,
                        "version": version,
                        "requested_models": train_result.get("requested_models", []),
                        "trained_models": train_result.get("trained_models", []),
                        "dependency_models": train_result.get("dependency_models", []),
                        "registry_dir": self.registry.base_dir,
                    },
                )
                manifest = {
                    "asset": asset,
                    "version": version,
                    "rows": int(len(payload)),
                    "registry_dir": self.registry.base_dir,
                    "merged_out": str(merged_out),
                    "model_state_out": str(model_state_out),
                    "requested_models": train_result.get("requested_models", []),
                    "trained_models": train_result.get("trained_models", []),
                    "dependency_models": train_result.get("dependency_models", []),
                    "metrics": metrics,
                    "governance": self._governance_status(metrics),
                }
                save_json(self.artifact_root / asset / "train_manifest.json", manifest)
                results[asset] = manifest
                continue

            if not isinstance(payload, Mapping):
                raise TypeError("TrainOrchestrator.run(...) expects asset payloads as DataFrames or mapping configs.")

            results[asset] = self.run_asset(
                asset=asset,
                tf_dir=payload.get("tf_dir"),
                features_csv=payload.get("features_csv"),
                labels_csv=payload.get("labels_csv"),
                features_out=payload.get("features_out"),
                labels_out=payload.get("labels_out"),
                merged_out=payload.get("merged_out"),
                manifest_out=payload.get("manifest_out"),
                model_state_out=payload.get("model_state_out"),
                models=payload.get("models"),
                resume=bool(payload.get("resume", True)),
                force_retrain=bool(payload.get("force_retrain", False)),
                force_rebuild_frame=bool(payload.get("force_rebuild_frame", False)),
            )
        return results

    def _collect_metrics(self, asset: str, version: str) -> Dict[str, Any]:
        model_names = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
        by_model: Dict[str, Any] = {}
        cv_scores = []
        for name in model_names:
            metrics_path = Path(self.registry.base_dir) / f"{asset}_{name}" / version / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                metrics = pd.read_json(metrics_path, typ="series").to_dict()
            except Exception:
                import json

                metrics = json.loads(metrics_path.read_text())
            by_model[name] = metrics
            cv = metrics.get("cv_score")
            if isinstance(cv, (int, float)):
                cv_scores.append(float(cv))

        summary = {
            "avg_cv_score": float(sum(cv_scores) / len(cv_scores)) if cv_scores else None,
            "specialists_with_metrics": sorted(by_model.keys()),
        }
        return {"summary": summary, "by_model": by_model}

    @staticmethod
    def _governance_status(metrics: Dict[str, Any]) -> Dict[str, Any]:
        available = metrics.get("summary", {}).get("specialists_with_metrics", [])
        return {
            "submitted": False,
            "reason": (
                "Trainer currently persists specialist CV metrics but not the full "
                "risk/calibration governance payload required for automatic promotion."
            ),
            "available_metrics": available,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-level training orchestration entrypoint.")
    parser.add_argument("--config-dir", default=default_conf_dir(__file__))
    parser.add_argument("--asset", default=None, help="Asset symbol, e.g. XBTUSD")
    parser.add_argument("--tf-dir", default=None, help="Directory containing {ASSET}_{15m,1h,6h,12h}.csv")
    parser.add_argument("--features", default=None, help="Prepared features CSV")
    parser.add_argument("--labels", default=None, help="Prepared labels CSV or merged feature+label CSV")
    parser.add_argument("--features-out", default=None, help="Optional built features output")
    parser.add_argument("--labels-out", default=None, help="Optional generated labels output")
    parser.add_argument("--merged-out", default="artifacts/train/latest/training_frame.csv", help="Merged training frame output")
    parser.add_argument("--manifest-out", default="artifacts/train/latest/train_manifest.json", help="Training manifest output")
    parser.add_argument("--model-state-out", default="artifacts/train/latest/model_state.json", help="Model state output")
    parser.add_argument("--models", default=None, help="Comma-separated model list, e.g. liq_flow,flow_1h,meta_model")
    parser.add_argument("--model-registry", default=None, help="Override model registry directory")
    parser.add_argument("--artifact-root", default="artifacts/train/latest", help="Default artifact root for orchestrator outputs")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint/manifest resume logic.")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining even if manifest is reusable.")
    parser.add_argument("--force-rebuild-frame", action="store_true", help="Force rebuilding merged training frame from features/labels.")
    return parser.parse_args()


@runtime_logged("Training orchestrator runtime")
def main() -> None:
    args = parse_args()
    orchestrator = TrainOrchestrator(
        conf_dir=args.config_dir,
        model_registry=args.model_registry,
        artifact_root=args.artifact_root,
    )
    asset = default_asset(orchestrator.cfg, args.asset)
    manifest = orchestrator.run_asset(
        asset=asset,
        tf_dir=args.tf_dir,
        features_csv=args.features,
        labels_csv=args.labels,
        features_out=args.features_out,
        labels_out=args.labels_out,
        merged_out=args.merged_out,
        manifest_out=args.manifest_out,
        model_state_out=args.model_state_out,
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        resume=not args.no_resume,
        force_retrain=args.force_retrain,
        force_rebuild_frame=args.force_rebuild_frame,
    )
    console_stage(
        "Training complete",
        f"asset={manifest['asset']} version={manifest['version']} trained={', '.join(manifest['trained_models'])}",
        status="ok",
    )
    LOG.info("[TrainOrchestrator] Complete asset=%s version=%s", manifest["asset"], manifest["version"])


if __name__ == "__main__":
    main()
