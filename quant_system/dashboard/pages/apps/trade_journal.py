import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("trade_journal")


# ------------------------------------------------------------
# Load journal data
# ------------------------------------------------------------
def _load_journal():
    base = Path.cwd() / "backtest_outputs"
    trades_file = base / "trades.csv"
    reasoning_file = base / "reasoning.json"

    if not trades_file.exists():
        return None, None

    trades = pd.read_csv(trades_file)

    reasoning = {}
    if reasoning_file.exists():
        with open(reasoning_file, "r") as f:
            reasoning = json.load(f)

    return trades, reasoning


# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
def _render_filters(trades):
    st.sidebar.markdown("### Filters")

    sides = st.sidebar.multiselect(
        "Side",
        ["long", "short"],
        default=["long", "short"],
    )

    results = st.sidebar.multiselect(
        "Outcome",
        ["win", "loss"],
        default=["win", "loss"],
    )

    sessions = st.sidebar.multiselect(
        "Session",
        sorted(trades["session"].dropna().unique().tolist()),
        default=trades["session"].dropna().unique().tolist(),
    )

    regimes = st.sidebar.multiselect(
        "Regime",
        sorted(trades["regime"].dropna().unique().tolist()),
        default=trades["regime"].dropna().unique().tolist(),
    )

    start_date = st.sidebar.date_input("Start Date", trades["entry_time"].min())
    end_date = st.sidebar.date_input("End Date", trades["entry_time"].max())

    moonshots = st.sidebar.checkbox("Only Moonshots (≥5R)", value=False)

    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])

    df = df[
        (df["side"].isin(sides))
        & (df["result"].isin(results))
        & (df["session"].isin(sessions))
        & (df["regime"].isin(regimes))
        & (df["entry_time"] >= pd.Timestamp(start_date))
        & (df["entry_time"] <= pd.Timestamp(end_date))
    ]

    if moonshots:
        df = df[df["realized_r"] >= 5.0]

    return df


# ------------------------------------------------------------
# Trade Screenshot Panel
# ------------------------------------------------------------
def _chart_snapshot(time):
    """
    Injects JS command to make the TradingView chart jump to
    the selected trade's timestamp.
    """
    js = f"""
        <script>
            if (window.tv_chart_jump) {{
                window.tv_chart_jump({int(time)});
            }}
        </script>
    """
    st.markdown(js, unsafe_allow_html=True)


# ------------------------------------------------------------
# Reasoning Tree Renderer
# ------------------------------------------------------------
def _reasoning_tree(reason):
    if not reason:
        st.info("No reasoning data for this trade.")
        return

    st.markdown(
        """
        <style>
        .tree-box {
            background:#111;
            border:1px solid #333;
            padding:10px;
            border-radius:6px;
            margin-bottom:10px;
        }
        .tree-key { font-weight:600; color:#cce3ff; }
        .tree-val { color:#ddd; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for k, v in reason.items():
        st.markdown(
            f"""
            <div class="tree-box">
                <div class="tree-key">{k}</div>
                <div class="tree-val">{v}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# Main Page Renderer
# ------------------------------------------------------------
def render_journal(theme_choice, model_version):
    LOG.info("Rendering Trade Journal")

    st.markdown(
        """
        <h1>Trade Journal</h1>
        <span style="color:#888;">TradeZella-style Review • Reasoning • Snapshots • Filters</span>
        <hr style="margin-top:12px;margin-bottom:20px;opacity:0.25;">
        """,
        unsafe_allow_html=True,
    )

    trades, reasoning = _load_journal()

    if trades is None:
        st.warning("No trades found. Run a backtest first.")
        return

    # Side filters
    filtered = _render_filters(trades)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trades", len(filtered))
    col2.metric("Win Rate", f"{filtered['result'].eq('win').mean()*100:.2f}%")
    col3.metric("Avg R", f"{filtered['realized_r'].mean():.2f}")
    col4.metric("PnL", f"{filtered['pnl'].sum():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Master table
    st.subheader("Trades")
    st.dataframe(
        filtered[
            [
                "entry_time",
                "exit_time",
                "side",
                "result",
                "evr",
                "conf",
                "realized_r",
                "pnl",
                "session",
                "regime",
            ]
        ],
        height=300,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Trade Detail Viewer
    st.subheader("Trade Details")

    trade_ids = filtered.index.tolist()
    if not trade_ids:
        st.info("No trades match the filters.")
        return

    selected_id = st.selectbox("Select Trade", trade_ids)

    trade = filtered.loc[selected_id]
    ts = pd.to_datetime(trade["entry_time"]).timestamp()

    # Left = chart snapshot | Right = reasoning tree
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### Chart Snapshot")
        _chart_snapshot(ts)

    with right:
        st.markdown("### Reasoning Tree")
        tr_id = str(trade["trade_id"]) if "trade_id" in trade else str(selected_id)
        reason = reasoning.get(tr_id, {})
        _reasoning_tree(reason)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    csv = filtered.to_csv(index=False)
    st.download_button("Export Filtered Trades (CSV)", csv, "trades_filtered.csv")
