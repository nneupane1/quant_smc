"""
Forward Monitor Dashboard
------------------------

This Streamlit page provides a live view into the forward testing or live
trading environment.  It is intended to mirror the functionality of the
`forward_test` cockpit but with a lighter footprint suitable for production
monitoring.  It reads snapshots from a directory of JSON files (typically
``forward_outputs``) and displays current account state, open trades,
performance metrics and basic risk indicators.  The page can be extended
to subscribe to a real‑time data source (e.g. websockets) but uses simple
file polling by default to remain self‑contained.

The design follows a clear separation of concerns: data loading functions
encapsulate file I/O, metric computation functions derive numbers from
raw snapshots, and rendering functions build Streamlit components.  Inline
comments document each step to facilitate maintenance and future
enhancements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import altair as alt


@dataclass
class ForwardSnapshot:
    """Container for a single forward test snapshot."""

    timestamp: pd.Timestamp
    equity: float
    locked_profit: float
    free_capital: float
    open_risk: float
    confluence: float
    evr: float
    hazard: float
    regime: str

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ForwardSnapshot":
        """
        Construct a snapshot instance from a raw dictionary.  Keys
        not found in the dictionary default to zero or empty strings.
        """
        ts = pd.to_datetime(data.get("timestamp"))
        return cls(
            timestamp=ts,
            equity=float(data.get("equity", 0.0)),
            locked_profit=float(data.get("locked_profit", 0.0)),
            free_capital=float(data.get("free_capital", 0.0)),
            open_risk=float(data.get("open_risk", 0.0)),
            confluence=float(data.get("confluence", 0.0)),
            evr=float(data.get("evr", 0.0)),
            hazard=float(data.get("hazard", 0.0)),
            regime=str(data.get("regime", "")),
        )


def load_latest_snapshot(directory: Path) -> Optional[ForwardSnapshot]:
    """
    Load the most recent snapshot file from a directory.

    Snapshots are assumed to be JSON files with a timestamp in their
    filename (e.g., ``snapshot_20250101T120000.json``).  If no snapshot
    exists, returns ``None``.

    Parameters
    ----------
    directory : Path
        Path to the directory containing snapshot JSON files.

    Returns
    -------
    Optional[ForwardSnapshot]
        The latest snapshot or ``None`` if none are available.
    """
    if not directory.exists() or not directory.is_dir():
        return None
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        return None
    latest_file = json_files[-1]
    try:
        with latest_file.open("r") as f:
            data = json.load(f)
        return ForwardSnapshot.from_dict(data)
    except Exception:
        return None


def load_open_trades(path: Path) -> pd.DataFrame:
    """
    Load the current open trades from a CSV file.

    The CSV should contain at least ``symbol``, ``side``, ``entry_price``
    and ``risk`` columns.  Additional columns are preserved for display.
    """
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path)
    return trades


def render_account_overview(snapshot: ForwardSnapshot) -> None:
    """
    Render a row of KPI cards summarising account health.
    """
    if snapshot is None:
        st.info("No snapshot available.")
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity", f"${snapshot.equity:,.2f}")
    col2.metric("Locked Profit", f"${snapshot.locked_profit:,.2f}")
    col3.metric("Free Capital", f"${snapshot.free_capital:,.2f}")
    col4.metric("Open Risk", f"${snapshot.open_risk:,.2f}")
    col5.metric("Regime", snapshot.regime.title())


def render_signals(snapshot: ForwardSnapshot) -> None:
    """
    Render confluence, EVR and hazard values as gauges or bars.
    """
    if snapshot is None:
        return
    st.subheader("Signals")
    col1, col2, col3 = st.columns(3)
    col1.metric("Confluence", f"{snapshot.confluence:.2f}")
    col2.metric("EVR", f"{snapshot.evr:.2f}")
    col3.metric("Hazard", f"{snapshot.hazard:.2f}")


def render_open_trades(trades: pd.DataFrame) -> None:
    """
    Render a table of currently open trades.
    """
    st.subheader("Open Trades")
    if trades.empty:
        st.info("No open trades.")
        return
    display_cols = [
        "symbol",
        "side",
        "entry_price",
        "size",
        "stop",
        "target",
        "pnl",
    ]
    cols = [c for c in display_cols if c in trades.columns]
    styled = trades[cols].style.apply(
        lambda s: ["background-color:#00C853" if v > 0 else "background-color:#D50000" if v < 0 else "" for v in s]
        if s.name == "pnl"
        else ["" for _ in s],
        axis=0,
    )
    st.dataframe(styled, use_container_width=True)


def render_forward_monitor() -> None:
    """
    Main entrypoint for the forward monitor Streamlit page.

    This function orchestrates loading of the latest forward snapshot and open
    trades, then renders various panels to display account information,
    signal values and open positions.  The update interval can be adjusted
    via query parameters to suit the deployment environment.
    """
    st.set_page_config(page_title="Forward Monitor", layout="wide")
    st.title("Forward Monitor")

    # Determine directories from query parameters or default values
    params = st.experimental_get_query_params()
    snap_dir = Path(params.get("snapshot_dir", ["forward_outputs/snapshots"])[0])
    trades_file = Path(params.get("trades_file", ["forward_outputs/open_trades.csv"])[0])

    # Load data
    snapshot = load_latest_snapshot(snap_dir)
    trades = load_open_trades(trades_file)

    # Render panels
    render_account_overview(snapshot)
    render_signals(snapshot)
    render_open_trades(trades)

    # Optionally display a performance chart if a history of snapshots exists
    history_dir = Path(params.get("history_dir", ["forward_outputs/history"])[0])
    if history_dir.exists():
        # Concatenate all snapshot files into a DataFrame for plotting
        rows: List[Dict[str, any]] = []
        for file in sorted(history_dir.glob("*.json")):
            try:
                with file.open("r") as f:
                    data = json.load(f)
                snap = ForwardSnapshot.from_dict(data)
                rows.append(asdict(snap))
            except Exception:
                continue
        if rows:
            df_history = pd.DataFrame(rows)
            st.subheader("Equity History")
            line = (
                alt.Chart(df_history)
                .mark_line(color="#1E88E5")
                .encode(x="timestamp:T", y="equity:Q")
                .properties(height=300)
            )
            st.altair_chart(line, use_container_width=True)


# Allow execution via ``streamlit run forward_monitor.py``
if __name__ == "__main__":
    render_forward_monitor()
