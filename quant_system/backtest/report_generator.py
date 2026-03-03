"""
High-level backtest reporting helpers.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from quant_system.backtest.visuals.report_builder import build_report as build_report_artifacts
from quant_system.backtest.visuals.report_generator import render_report


def generate_backtest_artifacts(
    result: Dict,
    output_dir: Path,
    *,
    candles: Optional[pd.DataFrame] = None,
    smc_features: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
):
    trades = result.get("trades", pd.DataFrame())
    metrics = result.get("metrics", {})
    equity_df = result.get("equity_curve")
    execution_log = result.get("execution_log")
    return build_report_artifacts(
        trades,
        output_dir,
        metrics=metrics,
        equity_df=equity_df,
        execution_log=execution_log,
        candles=candles,
        smc_features=smc_features,
        starting_equity=starting_equity,
    )


def render_backtest_report(
    trades: pd.DataFrame,
    *,
    equity_df: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
    explain_model: Optional[str] = None,
    registry_dir: str = "models",
):
    return render_report(
        trades,
        equity_df=equity_df,
        starting_equity=starting_equity,
        explain_model=explain_model,
        registry_dir=registry_dir,
    )
