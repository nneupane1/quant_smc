"""
report_page.py
Streamlit wrapper for building and viewing the institutional backtest report.

Features:
 - Loads the backtest artifacts already stored in session_state
 - Builds the final report via BacktestReportGenerator
 - Displays the report inside Streamlit
 - Allows export to HTML and PDF
"""

import streamlit as st
import os
from quant_system.utils.logger import get_logger
from dashboard.report.report_generator import BacktestReportGenerator

LOG = get_logger("report_page")

st.set_page_config(
    page_title="Backtest Report",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Institutional Backtest Report")

# ---------------------------------------------------------------------
# Validate backtest availability
# ---------------------------------------------------------------------
if "bt_artifacts" not in st.session_state:
    st.warning("No backtest is loaded. Please run a backtest first.")
    st.stop()

candles, smc, exec_log, model_bundle, config = st.session_state["bt_artifacts"]

df = candles.copy()
trades = exec_log.copy()

if "equity" not in df.columns:
    st.error("Backtester output missing df['equity']. Cannot build report.")
    st.stop()

# ---------------------------------------------------------------------
# Risk statistics (optional)
# ---------------------------------------------------------------------
risk_ref = st.session_state.get("risk_stats", {})

# ---------------------------------------------------------------------
# Report directory
# ---------------------------------------------------------------------
output_dir = "generated_reports"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------
# UI: Build report
# ---------------------------------------------------------------------
if st.button("Generate Full Report"):
    LOG.info("Building institutional-level report...")
    gen = BacktestReportGenerator(
        df=df,
        trades=trades,
        risk_stats=risk_ref,
        output_dir=output_dir
    )

    html_report = gen.build_html()
    st.session_state["report_html"] = html_report
    st.success("Report generated successfully.")

# ---------------------------------------------------------------------
# If report generated, show it
# ---------------------------------------------------------------------
if "report_html" in st.session_state:
    st.markdown("---")
    st.subheader("Interactive Report Preview")

    html_text = st.session_state["report_html"]

    # Embed HTML report
    st.components.v1.html(html_text, height=1200, scrolling=True)

    # File paths
    html_path = os.path.join(output_dir, "report.html")
    pdf_path = os.path.join(output_dir, "report.pdf")

    # Export PDF
    if st.button("Export PDF"):
        gen = BacktestReportGenerator(
            df=df,
            trades=trades,
            risk_stats=risk_ref,
            output_dir=output_dir
        )
        pdf = gen.export_pdf()
        if pdf:
            st.success(f"PDF exported → {pdf_path}")

    # Download buttons
    with open(html_path, "rb") as f:
        st.download_button(
            label="Download HTML Report",
            data=f,
            file_name="backtest_report.html",
            mime="text/html"
        )

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="backtest_report.pdf",
                mime="application/pdf"
            )
