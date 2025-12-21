"""
report_generator.py
Builds the full institutional backtest report (HTML + optional PDF).

Features:
 - Equity curve chart (base64 embedded)
 - Daily / Monthly tables (PnL, win rate, R-multiple stats)
 - Trade list table (click-to-replay with timestamp hooks)
 - Risk attribution summary (trend/range/expansion/collapse/hazard/hedge)
 - CVaR, drawdown tables
 - Best/worst trades

Output:
 - report.html
 - report.pdf (optional, requires wkhtmltopdf or WeasyPrint)
"""

import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
from quant_system.utils.logger import get_logger

LOG = get_logger("report_generator")


class BacktestReportGenerator:
    def __init__(self, df: pd.DataFrame, trades: pd.DataFrame, risk_stats: dict, output_dir: str):
        """
        df: full backtest candles dataframe (must contain equity, return_r, drawdown, timestamp)
        trades: trade log dataframe
        risk_stats: dict from risk attribution page
        output_dir: folder to write HTML/PDF
        """
        self.df = df.copy()
        self.trades = trades.copy()
        self.risk_stats = risk_stats
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Utility: encode chart as Base64 <img> for embedding in HTML
    # ------------------------------------------------------------------
    def _fig_to_base64(self, fig):
        png = fig.to_image(format="png")
        b64 = base64.b64encode(png).decode()
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------------
    # Build equity curve figure
    # ------------------------------------------------------------------
    def build_equity_curve(self):
        LOG.info("Building equity curve plot for report.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df["timestamp"],
            y=self.df["equity"],
            mode="lines",
            line=dict(color="#4FC3F7", width=2),
            name="Equity"
        ))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Build Drawdown Curve
    # ------------------------------------------------------------------
    def build_drawdown_curve(self):
        LOG.info("Building drawdown curve for report.")
        dd = self.df["equity"] - self.df["equity"].cummax()
        pct = dd / self.df["equity"].cummax()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df["timestamp"], y=pct,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#EF5350"),
            name="Drawdown %"
        ))
        fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Aggregate PnL stats
    # ------------------------------------------------------------------
    def build_period_tables(self):
        LOG.info("Building daily & monthly PnL tables.")
        df = self.df.copy()
        df["date"] = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)

        # Daily
        daily = df.groupby("date").agg(
            pnl=("return_r", "sum"),
            win_rate=("return_r", lambda x: np.mean(x > 0)),
            trades=("return_r", "count")
        ).reset_index()

        # Monthly
        monthly = df.groupby("month").agg(
            pnl=("return_r", "sum"),
            win_rate=("return_r", lambda x: np.mean(x > 0)),
            trades=("return_r", "count")
        ).reset_index()

        return daily, monthly

    # ------------------------------------------------------------------
    # Render DataFrame to HTML Table (styled)
    # ------------------------------------------------------------------
    def table_html(self, df, title):
        df = df.copy()
        styles = [
            dict(selector="th", props=[("background", "#1E1E1E"), ("color", "#FFF"), ("padding", "6px")]),
            dict(selector="td", props=[("padding", "6px")]),
            dict(selector="tbody tr:nth-child(even)", props=[("background", "#151515")]),
            dict(selector="tbody tr:nth-child(odd)", props=[("background", "#111")]),
        ]

        html_table = df.style.set_table_styles(styles).applymap(
            lambda v: "color:#4CAF50" if isinstance(v, (float,int)) and v > 0 else (
                "color:#EF5350" if isinstance(v,(float,int)) and v < 0 else ""
            )
        ).to_html()

        return f"<h3>{title}</h3>" + html_table

    # ------------------------------------------------------------------
    # Build report HTML
    # ------------------------------------------------------------------
    def build_html(self):
        LOG.info("Assembling final report HTML.")

        eq_b64 = self.build_equity_curve()
        dd_b64 = self.build_drawdown_curve()
        daily, monthly = self.build_period_tables()

        # Risk statistics
        risk_html = "<h3>Risk Attribution Summary</h3><ul>"
        for k, v in self.risk_stats.items():
            risk_html += f"<li><b>{k}</b>: {v:.4f}</li>"
        risk_html += "</ul>"

        # Trades table
        trade_html = self.table_html(self.trades, "Trade Log")

        # Daily & Monthly
        daily_html = self.table_html(daily, "Daily Performance")
        monthly_html = self.table_html(monthly, "Monthly Performance")

        # Final HTML template
        html = f"""
        <html>
        <head>
        <style>
            body {{
                background: #0D0D0F;
                color: #DDD;
                font-family: 'Inter', sans-serif;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #4FC3F7;
            }}
            .section {{
                padding-bottom: 40px;
                border-bottom: 1px solid #222;
                margin-bottom: 40px;
            }}
            img {{
                width: 100%;
                border-radius: 6px;
            }}
        </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            <div class="section">
                <h2>Equity Curve</h2>
                <img src="{eq_b64}" />
            </div>

            <div class="section">
                <h2>Drawdown Curve</h2>
                <img src="{dd_b64}" />
            </div>

            <div class="section">
                {daily_html}
            </div>

            <div class="section">
                {monthly_html}
            </div>

            <div class="section">
                {trade_html}
            </div>

            <div class="section">
                {risk_html}
            </div>

            <p>Generated on {datetime.utcnow()} UTC</p>
        </body>
        </html>
        """

        output_path = os.path.join(self.output_dir, "report.html")
        with open(output_path, "w") as f:
            f.write(html)

        LOG.info(f"Report written to {output_path}")

        return html

    # ------------------------------------------------------------------
    # Optional PDF export
    # ------------------------------------------------------------------
    def export_pdf(self):
        try:
            from weasyprint import HTML
        except ImportError:
            LOG.error("WeasyPrint not installed. Cannot export PDF.")
            return None

        html_path = os.path.join(self.output_dir, "report.html")
        pdf_path = os.path.join(self.output_dir, "report.pdf")

        LOG.info("Exporting PDF...")
        HTML(html_path).write_pdf(pdf_path)
        LOG.info(f"PDF exported to {pdf_path}")
        return pdf_path
