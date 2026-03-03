"""
SMC inspection helpers from replay/backtest artifacts.
"""

from typing import Dict

import pandas as pd


def summarize(trade_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if trade_df is None or trade_df.empty:
        return {"gate_reasons": pd.DataFrame(), "tiers": pd.DataFrame(), "legs": pd.DataFrame()}

    gate_reason_rows = []
    for _, row in trade_df.iterrows():
        reasons = row.get("gate_reasons", []) or []
        if isinstance(reasons, str):
            reasons = [reasons]
        for reason in reasons:
            gate_reason_rows.append({"trade_id": row.get("trade_id"), "reason": reason})

    gate_df = pd.DataFrame(gate_reason_rows)
    gate_summary = (
        gate_df.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)
        if not gate_df.empty
        else pd.DataFrame(columns=["reason", "count"])
    )

    tier_summary = (
        trade_df.groupby("tier")
        .agg(trades=("pnl", "count"), pnl_sum=("pnl", "sum"), avg_r=("r", "mean"))
        .reset_index()
        if "tier" in trade_df.columns
        else pd.DataFrame()
    )
    leg_summary = (
        trade_df.groupby("leg")
        .agg(trades=("pnl", "count"), pnl_sum=("pnl", "sum"), avg_r=("r", "mean"))
        .reset_index()
        if "leg" in trade_df.columns
        else pd.DataFrame()
    )
    return {"gate_reasons": gate_summary, "tiers": tier_summary, "legs": leg_summary}
