"""
Composable Streamlit report renderer for backtest outputs.
"""

from typing import Optional

import pandas as pd

from quant_system.backtest.visuals.drawdown_plot import drawdown_chart
from quant_system.backtest.visuals.equity_curve import equity_curve_chart
from quant_system.backtest.visuals.heatmaps import regime_heatmap, session_heatmap, tier_heatmap
from quant_system.backtest.visuals.ml_explain import explain_trade
from quant_system.backtest.visuals.moonshots import moonshot_table
from quant_system.backtest.visuals.tier_breakdown import tier_breakdown


def render_report(
    trade_df: pd.DataFrame,
    *,
    equity_df: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
    explain_model: Optional[str] = None,
    registry_dir: str = "models",
):
    equity_curve_chart(trade_df, equity_df=equity_df, starting_equity=starting_equity, render=True)
    drawdown_chart(trade_df, equity_df=equity_df, starting_equity=starting_equity, render=True)
    session_heatmap(trade_df, render=True)
    regime_heatmap(trade_df, render=True)
    tier_heatmap(trade_df, render=True)
    tier_breakdown(trade_df, render=True)
    moonshot_table(trade_df, render=True)

    if explain_model and not trade_df.empty:
        feature_like = trade_df.select_dtypes(include=["number", "bool"]).head(1)
        explain_trade(feature_like, model_name=explain_model, registry_dir=registry_dir, render=True)
