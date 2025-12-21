import streamlit as st
import pandas as pd


def tier_breakdown(df: pd.DataFrame):
    df = df.copy()

    tbl = df.groupby("tier").agg(
        pnl_sum=("pnl", "sum"),
        trades=("pnl", "count"),
        win_rate=("result", lambda x: (x == "win").mean() * 100),
        avg_r=("realized_r", "mean"),
        evr_avg=("evr", "mean"),
        hazard_avg=("hazard_at_entry", "mean"),
    ).reset_index()

    st.subheader("Tier Breakdown (A+ / A / B)")
    st.dataframe(tbl, use_container_width=True)
