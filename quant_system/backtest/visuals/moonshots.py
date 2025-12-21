import streamlit as st
import pandas as pd


def moonshot_table(df: pd.DataFrame, threshold: float = 5.0):
    """
    Display all trades with realized R greater than or equal to the threshold.
    """
    ms = df[df["realized_r"] >= threshold] if not df.empty else pd.DataFrame()

    st.subheader(f"Moonshot Trades (≥{threshold}R)")
    st.dataframe(
        ms[["entry_time", "side", "realized_r", "pnl", "regime", "session"]] if not ms.empty else ms,
        use_container_width=True,
        height=350,
    )
