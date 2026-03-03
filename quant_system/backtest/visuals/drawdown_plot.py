from typing import Optional

import pandas as pd

from quant_system.backtest.visuals.equity_curve import prepare_equity_curve


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


def prepare_drawdown_curve(trade_df: pd.DataFrame, equity_df: Optional[pd.DataFrame] = None, starting_equity: float = 0.0) -> pd.DataFrame:
    eq = prepare_equity_curve(trade_df, equity_df=equity_df, starting_equity=starting_equity)
    if eq.empty:
        return pd.DataFrame(columns=["time", "dd"])
    out = eq.copy()
    out["dd"] = out["equity_after"] - out["equity_after"].cummax()
    return out[["time", "dd"]]


def drawdown_chart(
    trade_df: pd.DataFrame,
    equity_df: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
    render: bool = True,
):
    data = prepare_drawdown_curve(trade_df, equity_df=equity_df, starting_equity=starting_equity)
    if not render:
        return data

    st = _optional_streamlit()
    alt = _optional_altair()
    if st is None or alt is None:
        return data

    chart = (
        alt.Chart(data)
        .mark_area(color="#FF4D4D", opacity=0.45)
        .encode(
            x="time:T",
            y=alt.Y("dd:Q", title="Drawdown"),
        )
        .interactive()
    )
    st.subheader("Drawdown (Underwater) Chart")
    st.altair_chart(chart, use_container_width=True)
    return chart
