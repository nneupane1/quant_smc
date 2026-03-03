"""
Report Builder
Transforms backtest outputs into saved artifacts and an optional Streamlit view.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("report_builder")


def _optional_streamlit():
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def _color_pnl(val):
    if val > 0:
        return "color: #00FF99;"
    if val < 0:
        return "color: #FF4D4D;"
    return "color: white;"


def _daily_report(df: pd.DataFrame):
    out = df.copy()
    out["date"] = pd.to_datetime(out["entry_ts"]).dt.date
    return (
        out.groupby("date")
        .agg(
            pnl_sum=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100),
            avg_r=("r", "mean"),
            best_r=("r", "max"),
            worst_r=("r", "min"),
        )
        .reset_index()
    )


def _monthly_report(df: pd.DataFrame):
    out = df.copy()
    out["month"] = pd.to_datetime(out["entry_ts"]).dt.to_period("M")
    monthly = (
        out.groupby("month")
        .agg(
            pnl_sum=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100),
            avg_r=("r", "mean"),
            best_r=("r", "max"),
            worst_r=("r", "min"),
        )
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly


def _summary(df: pd.DataFrame, starting_equity: float = 0.0) -> Dict[str, float]:
    pnl = df["pnl"].fillna(0.0)
    equity = starting_equity + pnl.cumsum()
    dd = equity - equity.cummax()
    return {
        "starting_equity": float(starting_equity),
        "ending_equity": float(equity.iloc[-1]) if not equity.empty else float(starting_equity),
        "total_pnl": float(pnl.sum()),
        "win_rate": float((df["pnl"] > 0).mean()) if not df.empty else 0.0,
        "max_drawdown": float(dd.min()) if not dd.empty else 0.0,
        "trades": int(len(df)),
    }


def launch_dashboard(trade_log_path: Path):
    st = _optional_streamlit()
    if st is None:
        raise RuntimeError("streamlit is required to launch the backtest dashboard.")

    if not trade_log_path.exists():
        raise FileNotFoundError(f"Trade log missing: {trade_log_path}")

    df = pd.read_csv(trade_log_path, parse_dates=["entry_ts", "exit_ts"])
    if df.empty:
        raise ValueError("Trade log is empty.")

    summary = _summary(df)
    daily = _daily_report(df)
    monthly = _monthly_report(df)

    st.markdown(
        """
        <h1>Backtest Results Dashboard</h1>
        <span style='color:#AAA;'>Summary • Daily Breakdown • Monthly Breakdown • Ledger</span>
        <hr style='opacity:0.2;margin-top:10px;'>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Ending Equity", f"{summary['ending_equity']:,.2f}")
    cols[1].metric("Total PnL", f"{summary['total_pnl']:,.2f}")
    cols[2].metric("Trades", f"{summary['trades']}")
    cols[3].metric("Win Rate", f"{summary['win_rate'] * 100:.2f}%")
    cols[4].metric("Max Drawdown", f"{summary['max_drawdown']:,.2f}")

    st.subheader("Daily Performance")
    st.dataframe(daily.style.applymap(_color_pnl, subset=["pnl_sum"]), height=300, use_container_width=True)

    st.subheader("Monthly Performance")
    st.dataframe(monthly.style.applymap(_color_pnl, subset=["pnl_sum"]), height=300, use_container_width=True)

    st.subheader("Trade Ledger")
    st.dataframe(df.style.applymap(_color_pnl, subset=["pnl"]), height=500, use_container_width=True)
    return {"summary": summary, "daily": daily, "monthly": monthly, "ledger": df}


def build_report(
    df: pd.DataFrame,
    save_path: Path,
    *,
    metrics: Optional[Dict] = None,
    equity_df: Optional[pd.DataFrame] = None,
    execution_log: Optional[pd.DataFrame] = None,
    candles: Optional[pd.DataFrame] = None,
    smc_features: Optional[pd.DataFrame] = None,
    starting_equity: float = 0.0,
):
    """
    Save backtest artifacts for downstream dashboard/replay use.
    """
    LOG.info("Building standalone report artifacts")
    save_path.mkdir(parents=True, exist_ok=True)

    daily = _daily_report(df) if not df.empty else pd.DataFrame()
    monthly = _monthly_report(df) if not df.empty else pd.DataFrame()
    summary = metrics or _summary(df, starting_equity=starting_equity)

    df.to_csv(save_path / "ledger.csv", index=False)
    daily.to_csv(save_path / "daily_report.csv", index=False)
    monthly.to_csv(save_path / "monthly_report.csv", index=False)
    with open(save_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if equity_df is not None:
        equity_df.to_csv(save_path / "equity_curve.csv", index=False)
    if execution_log is not None:
        execution_log.to_csv(save_path / "execution_log.csv", index=False)
    if candles is not None:
        candles.to_csv(save_path / "candles_15m.csv", index=False)
    if smc_features is not None:
        smc_features.to_csv(save_path / "smc_features.csv", index=False)

    LOG.info("Report artifacts saved.")
    return {
        "summary": save_path / "summary.json",
        "ledger": save_path / "ledger.csv",
        "daily": save_path / "daily_report.csv",
        "monthly": save_path / "monthly_report.csv",
        "equity_curve": save_path / "equity_curve.csv" if equity_df is not None else None,
        "execution_log": save_path / "execution_log.csv" if execution_log is not None else None,
        "candles": save_path / "candles_15m.csv" if candles is not None else None,
        "smc_features": save_path / "smc_features.csv" if smc_features is not None else None,
    }
