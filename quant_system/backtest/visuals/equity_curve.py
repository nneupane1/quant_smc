from typing import Optional

import pandas as pd


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def _optional_altair():
    try:
        import altair as alt

        return alt
    except Exception:
        return None


def prepare_equity_curve(trade_df: pd.DataFrame, equity_df: Optional[pd.DataFrame] = None, starting_equity: float = 0.0) -> pd.DataFrame:
    if equity_df is not None and not equity_df.empty:
        out = equity_df.copy()
        if "time" not in out.columns:
            if "timestamp" in out.columns:
                out["time"] = pd.to_datetime(out["timestamp"], errors="coerce")
            elif "dt" in out.columns:
                out["time"] = pd.to_datetime(out["dt"], errors="coerce")
        if "equity_after" not in out.columns and "equity" in out.columns:
            out["equity_after"] = out["equity"]
        return out[["time", "equity_after"]].dropna().reset_index(drop=True)

    df = trade_df.copy() if trade_df is not None else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["time", "equity_after"])

    ts_col = "exit_ts" if "exit_ts" in df.columns else "entry_ts"
    out = df.loc[pd.notna(df[ts_col]), [ts_col, "pnl"]].copy()
    out["time"] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.sort_values("time")
    out["equity_after"] = float(starting_equity) + out["pnl"].fillna(0.0).cumsum()
    return out[["time", "equity_after"]].dropna().reset_index(drop=True)


def equity_curve_chart(
    trade_df: pd.DataFrame,
    equity_df: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
    render: bool = True,
):
    data = prepare_equity_curve(trade_df, equity_df=equity_df, starting_equity=starting_equity)
    if not render:
        return data

    st = _optional_streamlit()
    alt = _optional_altair()
    if st is None or alt is None:
        return data

    chart = (
        alt.Chart(data)
        .mark_line(size=2, color="#3FA9F5")
        .encode(
            x=alt.X("time:T", axis=alt.Axis(title="Time")),
            y=alt.Y("equity_after:Q", axis=alt.Axis(title="Equity")),
        )
        .interactive()
    )
    st.subheader("Equity Curve")
    st.altair_chart(chart, use_container_width=True)
    return chart
