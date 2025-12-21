"""
dual_dashboard_adapter.py
Side-by-side dashboard projection for:
 • TV chart (split screen)
 • PnL curves
 • Equity panels
 • MPC panels
 • Trade logs
 • Confluence / EVR comparison
 • Hazard comparisons
"""

import streamlit as st


class DualDashboardAdapter:

    def __init__(self, dashboard_A, dashboard_B):
        self.A = dashboard_A
        self.B = dashboard_B

    def update_candles(self, data_A, data_B):
        """Push candles for both models."""
        if data_A:
            self.A.update_candles(data_A)
        if data_B:
            self.B.update_candles(data_B)

    def update_metrics(self, metrics_A, metrics_B):
        """PnL, equity, vault, drawdown."""
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Model A")
            self.A.update_metrics(metrics_A)

        with colB:
            st.subheader("Model B")
            self.B.update_metrics(metrics_B)

    def update_trade_logs(self, logs_A, logs_B):
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Trades — Model A")
            self.A.update_trade_table(logs_A)

        with colB:
            st.subheader("Trades — Model B")
            self.B.update_trade_table(logs_B)

    def update_reasoning(self, rA, rB):
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Reasoning — Model A")
            self.A.update_reasoning(rA)

        with colB:
            st.subheader("Reasoning — Model B")
            self.B.update_reasoning(rB)
