import pandas as pd


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def moonshot_table(df: pd.DataFrame, threshold: float = 5.0, render: bool = True):
    if df is None or df.empty:
        ms = pd.DataFrame()
    else:
        r_col = "r" if "r" in df.columns else "realized_r"
        ts_col = "entry_ts" if "entry_ts" in df.columns else "entry_time"
        session_col = "session" if "session" in df.columns else None
        regime_col = "regime" if "regime" in df.columns else None
        cols = [c for c in [ts_col, "side", r_col, "pnl", regime_col, session_col] if c]
        ms = df[df[r_col] >= threshold][cols].copy() if r_col in df.columns else pd.DataFrame()
        ms = ms.rename(columns={ts_col: "entry_time", r_col: "realized_r"})

    if not render:
        return ms

    st = _optional_streamlit()
    if st is None:
        return ms

    st.subheader(f"Moonshot Trades (≥{threshold}R)")
    st.dataframe(ms, use_container_width=True, height=350)
    return ms
