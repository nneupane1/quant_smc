"""
Backtest Report Dashboard
-------------------------

This Streamlit page generates a comprehensive backtest report from data stored in
the local ``backtest_outputs`` directory.  It loads summary metrics and trade
details, computes additional performance statistics such as equity curve,
drawdown and risk ratios, and renders interactive charts and tables to
facilitate detailed analysis of historical trading performance.

The implementation emphasises clear separation of concerns: data loading,
metric computation and user-interface rendering are encapsulated in
dedicated functions.  This modularity makes the code testable and easy to
extend for new metrics.  Extensive inline comments explain the purpose of
each function and the rationale behind calculations to ensure maintainability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


@dataclass
class BacktestSummary:
    """Container for summary statistics loaded from a JSON file."""

    metrics: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "BacktestSummary":
        """
        Load backtest summary metrics from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file containing summary metrics.

        Returns
        -------
        BacktestSummary
            A summary instance with the loaded metrics.
        """
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        with path.open("r") as f:
            data = json.load(f)
        return cls(metrics=data)



def load_trades(path: Path) -> pd.DataFrame:
    """
    Load trade data from a CSV file into a DataFrame.

  ----------
    path : Path
        Path to the CSV file containing trades.  The CSV is expected to have
        at least the following columns:

        - ``timestamp``: ISO‐8601 timestamp for trade entry.
        - ``pnl``: realised profit or loss in account currency.
        - ``r_mult``: realised R‐multiple (return normalised by stop size).

    Returns
    -------
    pandas.DataFrame
        DataFrame with loaded trades.  Timestamp is converted to ``datetime64``.
        Returns an empty DataFrame if the file does not exist.
    """
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path)
    # Convert timestamp column to datetime if present
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    return trades



def compute_equity(trades: pd.DataFrame, starting_capital: float = 10_000.0) -> pd.DataFrame:
    """
    Compute the cumulative equity curve and drawdown series from a trades DataFrame.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades DataFrame with at least the ``timestamp`` and 
