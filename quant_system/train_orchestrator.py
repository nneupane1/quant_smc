"""
train_orchestrator.py
Walk-forward training orchestration:
 1) Load feature/label frames per asset
 2) Optional preprocessing (scaling handled inside ModelTrainer calibration/HPO pipeline)
 3) Run ModelTrainer per asset (uses Optuna + TSCV)
 4) Persist metrics and governance decisions
"""

import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.training.model_trainer import ModelTrainer
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.model_ensemble.model_governor import ModelGovernor
from quant_system.utils.logger import get_logger

LOG = get_logger("train_orchestrator")


class TrainOrchestrator:
    def __init__(self, conf_dir: str = "quant_system/config"):
        self.cfg_loader = ConfigLoader(conf_dir)
        cfg = self.cfg_loader.load()

        models_root = cfg.get("paths", {}).get("model_registry", "artifacts/models")
        self.registry = ModelRegistry(models_root)
        self.governor = ModelGovernor(self.registry, cfg.get("models", {}).get("governance", {}))

        self.assets_cfg = cfg.get("assets", {})
        self.output_dir = Path(models_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, asset_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        asset_frames: {asset: feature+label dataframe} where df contains:
          - feature columns
          - label_liq_flow, label_bos_cont, label_momo, label_eop, label_edp
          - hazard_event, hazard_time, close
        Returns version IDs per asset.
        """
        trainer = ModelTrainer(self.cfg_loader, self.registry)
        versions = {}

        for asset, df in asset_frames.items():
            LOG.info(f"[Orchestrator] Training asset={asset}, rows={len(df)}")
            version = trainer.train_asset(df, asset)
            versions[asset] = version

        # Governance placeholder: approve if versions exist
        for asset, ver in versions.items():
            self.governor.submit(
                model_id=f"{asset}_{ver}",
                metrics={"pr_auc": 0.0, "brier": 1.0},  # replace with real evals
                risk={"max_dd": 0.0, "cvar95": 0.0},
                calib={"ece": 1.0},
            )

        return versions


if __name__ == "__main__":
    LOG.info("TrainOrchestrator entrypoint expects prebuilt feature/label frames.")
    # Example usage (pseudocode):
    # frames = {"BTCUSD": pd.read_csv("features_labels_btc.csv")}
    # orchestrator = TrainOrchestrator()
    # orchestrator.run(frames)
