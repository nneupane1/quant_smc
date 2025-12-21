"""
exporter.py
Exports a full standalone HTML replay movie:
 • OHLC animations
 • PnL curve animation
 • Trades
 • SMC overlays
 • Reasoning tree per bar
 • Timeline slider + auto-play
No external dependencies once exported.
"""

import json
import base64
from pathlib import Path
from quant_system.utils.logger import get_logger

LOG = get_logger("replay_exporter")


class ReplayExporter:

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def export(self, replay_data: dict):
        """
        replay_data = {
            "bars": [...]     # list of dicts per bar: o,h,l,c,dt, overlays
            "pnl":  [...]     # running pnl/equity list
            "trades": [...]   # executed trades
            "reasoning": {...} per dt
        }
        """
        LOG.info("Preparing replay export…")

        html = self._build_html(
            bars=replay_data["bars"],
            pnl=replay_data["pnl"],
            trades=replay_data["trades"],
            reasoning=replay_data["reasoning"]
        )

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        LOG.info(f"Replay exported successfully: {self.output_path}")

    # ------------------------------------------------------------------
    def _build_html(self, bars, pnl, trades, reasoning):
        LOG.info("Embedding JSON payload into template")

        data_bundle = {
            "bars": bars,
            "pnl": pnl,
            "trades": trades,
            "reasoning": reasoning
        }
        data_json = json.dumps(data_bundle)

        # Load template
        html_template = Path(
            "quant_system/replay_export/template.html"
        ).read_text()

        # Inject JS + CSS inline for standalone portability
        js = Path("quant_system/replay_export/replay.js").read_text()
        css = Path("quant_system/replay_export/styles.css").read_text()

        html = (
            html_template
            .replace("/*__CSS__*/", css)
            .replace("//__JS__", js)
            .replace("__DATA__", data_json)
        )
        return html
