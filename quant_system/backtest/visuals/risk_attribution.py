"""
Risk attribution helpers from the current trade ledger.
"""

from typing import Dict

import pandas as pd


def _group_summary(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(key)
        .agg(
            trades=("pnl", "count"),
            pnl_sum=("pnl", "sum"),
            avg_r=("r", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        )
        .reset_index()
        .sort_values("pnl_sum", ascending=False)
    )


def compute(trade_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if trade_df is None or trade_df.empty:
        return {"by_leg": pd.DataFrame(), "by_tier": pd.DataFrame(), "by_regime": pd.DataFrame(), "by_session": pd.DataFrame()}
    return {
        "by_leg": _group_summary(trade_df, "leg"),
        "by_tier": _group_summary(trade_df, "tier"),
        "by_regime": _group_summary(trade_df, "regime"),
        "by_session": _group_summary(trade_df, "session"),
    }
