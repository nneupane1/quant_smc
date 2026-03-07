from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "sklearn", "xgboost", "lightgbm", "optuna"))

import warnings
from pathlib import Path
import time

from pandas.errors import DtypeWarning

from quant_system.cli.common import load_or_build_features
from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.tuner import LabelEmpiricalTuner
from quant_system.utils.logger import console_stage, fmt_seconds

ASSET = "BTCUSD"
CONFIG_DIR = "quant_system/config"
FEATURES_CSV = f"artifacts/features/{ASSET}/{ASSET}_features.csv"
TF_DIR = "data/tf"
OUTPUT_DIR = f"artifacts/label_tuning/{ASSET}"
AUTO_PROMOTE = True
MIN_IMPROVEMENT = 0.01
RESUME = True


def main() -> None:
    started_at = time.perf_counter()
    completed = False
    try:
        warnings.filterwarnings("ignore", category=DtypeWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"Could not infer format, so each element will be parsed individually.*",
            category=UserWarning,
        )
        loader = ConfigLoader(CONFIG_DIR)
        features_df = load_or_build_features(
            loader,
            asset=ASSET,
            features_csv=FEATURES_CSV if Path(FEATURES_CSV).exists() else None,
            tf_dir=TF_DIR,
            features_out=FEATURES_CSV,
        )
        tuner = LabelEmpiricalTuner(loader)
        tuner.tune(
            features_df,
            output_dir=OUTPUT_DIR,
            auto_promote=AUTO_PROMOTE,
            min_improvement=MIN_IMPROVEMENT,
            resume=RESUME,
        )
        completed = True
        console_stage(
            "Label tuning complete",
            f"reports={OUTPUT_DIR}",
            status="ok",
        )
    finally:
        console_stage(
            "Label tuning runtime",
            f"elapsed={fmt_seconds(time.perf_counter() - started_at)}",
            status="ok" if completed else "warn",
        )


if __name__ == "__main__":
    main()
