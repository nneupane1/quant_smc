import streamlit as st
import pandas as pd
import altair as alt
from quant_system.utils.logger import get_logger

LOG = get_logger("equity_curve")


def equity_curve_chart(df: pd.DataFrame):
    """
    Renders an interactive equity curve using Altair.
    Smooth, zoomable, Bloomberg-style gradient.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['entry_time'])

    line = (
        alt.Chart(df)
        .mark_line(size=2, color="#3FA9F5")
        .encode(
            x=alt.X("time:T", axis=alt.Axis(title="Time")),
            y=alt.Y("equity_after:Q", axis=alt.Axis(title="Equity")),
        )
    )

    area = (
        alt.Chart(df)
        .mark_area(
            opacity=0.25,
            color="#3FA9F5"
        )
        .encode(
            x="time:T",
            y="equity_after:Q"
        )
    )

    chart = (area + line).interactive()
    st.subheader("Equity Curve")
    st.altair_chart(chart, use_container_width=True)
