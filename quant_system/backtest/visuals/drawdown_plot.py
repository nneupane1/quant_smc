import streamlit as st
import pandas as pd
import altair as alt
from quant_system.utils.logger import get_logger

LOG = get_logger("drawdown_plot")


def drawdown_chart(df: pd.DataFrame):
    """
    Underwater drawdown chart (Bloomberg style).
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['entry_time'])
    df['dd'] = df['equity_after'] - df['equity_after'].cummax()

    chart = (
        alt.Chart(df)
        .mark_area(color="#FF4D4D", opacity=0.45)
        .encode(
            x="time:T",
            y=alt.Y("dd:Q", title="Drawdown"),
        )
        .interactive()
    )

    st.subheader("Drawdown (Underwater) Chart")
    st.altair_chart(chart, use_container_width=True)
