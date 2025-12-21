"""
trade_summary_panel.py
Multi-Asset Bloomberg-style trade summary dashboard section.

Panels rendered:
 • Equity & Locked Vault
 • Daily / Weekly / Monthly PnL tables
 • Multi-asset PnL heatmap
 • Live trade tape
 • Open positions board
 • Strategy state board (cooling, DD, vault, risk, hazard)
 • Performance KPIs

Supports:
 • Instant updates via Streamlit states
 • JSON bridge for JS widgets
 • Multi-asset switching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


class TradeSummaryPanel:

    def __init__(self):
        if "trade_tape" not in st.session_state:
            st.session_state["trade_tape"] = []

        if "pnl_records" not in st.session_state:
            st.session_state["pnl_records"] = pd.DataFrame(columns=[
                "dt", "asset", "pnl", "r_multiple", "tier"
            ])

        if "open_positions" not in st.session_state:
            st.session_state["open_positions"] = {}

        if "equity" not in st.session_state:
            st.session_state["equity"] = 0.0

        if "locked_profit" not in st.session_state:
            st.session_state["locked_profit"] = 0.0

        if "max_drawdown" not in st.session_state:
            st.session_state["max_drawdown"] = 0.0

        if "cooling" not in st.session_state:
            st.session_state["cooling"] = False

    # ------------------------------------------------------------------
    # RENDER MAIN PANEL
    # ------------------------------------------------------------------
    def render(self):
        self._render_equity_panel()
        self._render_multi_asset_heatmap()
        self._render_daily_table()
        self._render_monthly_table()
        self._render_open_positions()
        self._render_trade_tape()
        self._render_strategy_state()

    # ------------------------------------------------------------------
    def push_trade(self, trade_info: dict):
        st.session_state["trade_tape"].append(trade_info)

        if "pnl" in trade_info and trade_info["pnl"] is not None:
            df = st.session_state["pnl_records"]
            st.session_state["pnl_records"] = pd.concat([df, pd.DataFrame([{
                "dt": trade_info["dt"],
                "asset": trade_info["asset"],
                "pnl": trade_info["pnl"],
                "r_multiple": trade_info.get("r_multiple", None),
                "tier": trade_info.get("tier", None)
            }])], ignore_index=True)

    # ------------------------------------------------------------------
    def update_state(self, info: dict):
        st.session_state["equity"] = info.get("equity", st.session_state["equity"])
        st.session_state["locked_profit"] = info.get("locked_profit", st.session_state["locked_profit"])
        st.session_state["max_drawdown"] = info.get("max_drawdown", st.session_state["max_drawdown"])
        st.session_state["cooling"] = info.get("cooling", st.session_state["cooling"])
        st.session_state["open_positions"] = info.get("open_positions", st.session_state["open_positions"])

    # ------------------------------------------------------------------
    def _render_equity_panel(self):
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"{st.session_state['equity']:.2f}")
        c2.metric("Locked Profit", f"{st.session_state['locked_profit']:.2f}")
        c3.metric("Max Drawdown", f"{st.session_state['max_drawdown']*100:.2f}%")

    # ------------------------------------------------------------------
    def _render_multi_asset_heatmap(self):
        df = st.session_state["pnl_records"]
        if df.empty:
            return

        heat = df.groupby("asset")["pnl"].sum().reset_index()
        fig = px.bar(
            heat,
            x="asset",
            y="pnl",
            color="pnl",
            color_continuous_scale="RdYlGn",
            title="Multi-Asset PnL Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    def _render_daily_table(self):
        df = st.session_state["pnl_records"]
        if df.empty:
            return

        df["date"] = pd.to_datetime(df["dt"]).dt.date
        daily = df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
        daily.columns = ["Date", "PnL", "#Trades"]

        st.subheader("Daily PnL")
        st.dataframe(daily.style.applymap(self._color_pnl, subset=["PnL"]))

    # ------------------------------------------------------------------
    def _render_monthly_table(self):
        df = st.session_state["pnl_records"]
        if df.empty:
            return

        df["month"] = pd.to_datetime(df["dt"]).dt.to_period("M")
        monthly = df.groupby("month")["pnl"].agg(["sum", "count"]).reset_index()
        monthly.columns = ["Month", "PnL", "#Trades"]

        st.subheader("Monthly PnL")
        st.dataframe(monthly.style.applymap(self._color_pnl, subset=["PnL"]))

    # ------------------------------------------------------------------
    def _render_open_positions(self):
        pos = st.session_state["open_positions"]
        if not pos:
            st.write("No open positions.")
            return

        rows = []
        for tid, p in pos.items():
            rows.append({
                "Trade ID": tid,
                "Asset": p["asset"],
                "Side": p["side"],
                "Qty": p["qty"],
                "Entry": p["entry"],
                "PnL": p["pnl"],
                "Risk": p["risk"],
                "Leverage": p["leverage"],
                "Hedge": p["hedge_ratio"]
            })

        df = pd.DataFrame(rows)
        st.subheader("Open Positions")
        st.dataframe(df.style.applymap(self._color_pnl, subset=["PnL"]))

    # ------------------------------------------------------------------
    def _render_trade_tape(self):
        st.subheader("Trade Tape")

        tape = st.session_state["trade_tape"][-250:]  # last 250 trades in view

        rows = []
        for t in reversed(tape):
            rows.append({
                "DT": t["dt"],
                "Asset": t["asset"],
                "Side": t.get("side"),
                "Tier": t.get("tier"),
                "Entry/Exit": t.get("pnl", "open"),
                "R-Mult": t.get("r_multiple")
            })

        df = pd.DataFrame(rows)
        st.dataframe(df.style.applymap(self._color_entry_exit, subset=["Entry/Exit"]))

    # ------------------------------------------------------------------
    def _render_strategy_state(self):
        st.subheader("Strategy State")

        c1, c2, c3, c4 = st.columns(4)

        cooling = "Active" if st.session_state["cooling"] else "Off"
        c1.metric("Cooling", cooling)

        c2.metric("Vault", f"{st.session_state['locked_profit']:.2f}")
        c3.metric("Equity", f"{st.session_state['equity']:.2f}")
        c4.metric("# Open Trades", len(st.session_state["open_positions"]))

    # ------------------------------------------------------------------
    @staticmethod
    def _color_pnl(val):
        try:
            v = float(val)
            return "color: #00d000" if v > 0 else "color: #ff4040"
        except:
            return ""

    @staticmethod
    def _color_entry_exit(val):
        try:
            v = float(val)
            return "color: #00ff00" if v > 0 else "color: #ff0000"
        except:
            return ""
