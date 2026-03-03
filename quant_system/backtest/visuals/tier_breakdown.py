import pandas as pd


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def tier_breakdown(df: pd.DataFrame, render: bool = True):
    if df is None or df.empty or "tier" not in df.columns:
        tbl = pd.DataFrame()
    else:
        hazard_col = "hazard_entry" if "hazard_entry" in df.columns else "hazard_at_entry"
        r_col = "r" if "r" in df.columns else "realized_r"
        tbl = (
            df.groupby("tier")
            .agg(
                pnl_sum=("pnl", "sum"),
                trades=("pnl", "count"),
                win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                avg_r=(r_col, "mean"),
                evr_avg=("evr", "mean"),
                hazard_avg=(hazard_col, "mean"),
            )
            .reset_index()
        )

    if not render:
        return tbl

    st = _optional_streamlit()
    if st is None:
        return tbl

    st.subheader("Tier Breakdown (A+ / A / B)")
    st.dataframe(tbl, use_container_width=True)
    return tbl
