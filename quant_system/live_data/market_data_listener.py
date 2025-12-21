"""
market_data_listener.py
Multi-asset real-time market data ingestion system.

• Connects to Kraken WebSocket (public OHLC feed)
• Subscribes to all assets in assets.yaml
• Streams live 1m candles
• Emits closed TF bars to ForwardEngine
• Updates TV chart and dashboard instantly
• Reconnect-safe, fault-tolerant
"""

import json
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Any

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

from quant_system.live_data.quote_state import QuoteState
from quant_system.live_data.tf_builder import TFBuilder


LOG = get_logger("market_data_listener")


class MarketDataListener:
    """Runs an async Kraken WS listener and produces TF bars."""

    def __init__(self, cfg: ConfigLoader, forward_engine=None, dashboard=None):
        self.cfg = cfg
        self.assets_cfg = cfg.load_yaml("assets.yaml")
        self.symbols = self.assets_cfg["assets"]["enabled"]

        self.forward = forward_engine
        self.dashboard = dashboard

        self.state = {sym: QuoteState() for sym in self.symbols}
        self.tf = {sym: TFBuilder() for sym in self.symbols}

        self.ws_url = "wss://ws.kraken.com"

    # --------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------
    async def run(self):
        while True:
            try:
                await self._connect_and_stream()
            except Exception as e:
                LOG.error(f"[MarketData] Stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    # --------------------------------------------------------------
    # CONNECT AND SUBSCRIBE
    # --------------------------------------------------------------
    async def _connect_and_stream(self):
        LOG.info("[MarketData] Connecting to Kraken WebSocket...")

        async with websockets.connect(self.ws_url, max_queue=None) as ws:
            await self._subscribe(ws)
            LOG.info("[MarketData] Subscribed. Awaiting data...")

            async for msg in ws:
                self._handle_message(msg)

    # --------------------------------------------------------------
    # SUBSCRIBE TO MULTI-ASSET OHLC
    # --------------------------------------------------------------
    async def _subscribe(self, ws):
        pairs = [self.assets_cfg["assets"]["metadata"][sym]["kraken_pair"] for sym in self.symbols]

        req = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ohlc", "interval": 1}
        }

        LOG.info(f"[MarketData] Subscribing: {pairs}")
        await ws.send(json.dumps(req))

    # --------------------------------------------------------------
    # HANDLE KRAKEN WS MESSAGE
    # --------------------------------------------------------------
    def _handle_message(self, msg: str):
        try:
            data = json.loads(msg)
        except:
            return

        # Ignore events
        if isinstance(data, dict):
            return

        # OHLC message format:
        # [channelID, [open,high,low,close,vwap,volume,count], "pair", {info...}]
        if len(data) < 3:
            return

        pair = data[3].get("pair")
        ohlc = data[1]

        # Find asset from kraken pair
        asset = None
        for sym, meta in self.assets_cfg["assets"]["metadata"].items():
            if meta["kraken_pair"] == pair:
                asset = sym
                break
        if not asset:
            return

        ts = float(ohlc[1])
        candle = {
            "dt": datetime.utcfromtimestamp(ts),
            "timestamp": int(ts),
            "open": float(ohlc[2]),
            "high": float(ohlc[3]),
            "low": float(ohlc[4]),
            "close": float(ohlc[5]),
            "volume": float(ohlc[7]),
            "asset": asset,
        }

        self.state[asset].push_1m(candle)

        # Dashboard real-time chart update
        if self.dashboard:
            self.dashboard.update_candles({asset: candle})

        # TF aggregation
        emits = self.tf[asset].push_1m(candle)

        for tf, bar in emits.items():
            if self.forward:
                self.forward.on_bar(asset, bar)

            if self.dashboard:
                self.dashboard.update_tf_bar(asset, tf, bar)
