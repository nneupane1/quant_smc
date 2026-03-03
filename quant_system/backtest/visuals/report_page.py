"""
Thin report page wrapper.
"""

from typing import Optional

import pandas as pd

from quant_system.backtest.visuals.report_generator import render_report


def render(
    trade_df: pd.DataFrame,
    *,
    equity_df: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
    explain_model: Optional[str] = None,
    registry_dir: str = "models",
):
    return render_report(
        trade_df,
        equity_df=equity_df,
        starting_equity=starting_equity,
        explain_model=explain_model,
        registry_dir=registry_dir,
    )
