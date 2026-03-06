"""Kraken WebSocket-first market-data listener with REST fallback."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import websockets

from quant_system.config.config_loader import ConfigLoader
from quant_system.live.kraken_live_client import KrakenLiveClient
from quant_system.live_data.live_feature_enricher import LiveFeatureEnricher
from quant_system.live_data.quote_state import QuoteState
from quant_system.live_data.tf_builder import TFBuilder
from quant_system.utils.logger import get_logger

LOG = get_logger("market_data_listener")


class MarketDataListener:
    """Kraken WS listener that emits closed 1m and higher-timeframe bars."""

    def __init__(self, cfg: ConfigLoader, forward_engine=None, dashboard=None):
        self.cfg_loader = cfg
        self.cfg = cfg.load()
        assets_cfg = self.cfg.get("assets", {})
        self.symbols = list(assets_cfg.get("enabled", []))
        self.metadata = assets_cfg.get("metadata", {})

        self.forward = forward_engine
        self.dashboard = dashboard

        self.state = {sym: QuoteState() for sym in self.symbols}
        self.tf = {sym: TFBuilder() for sym in self.symbols}
        self.enricher = LiveFeatureEnricher(cfg)
        self.strict_gate_mode = bool(self.cfg.get("execution", {}).get("gates", {}).get("strict_mode", False))
        self.running_1m: Dict[str, Optional[dict]] = {sym: None for sym in self.symbols}
        self.rest_clients = self._build_rest_clients()

        self.ws_url = "wss://ws.kraken.com"
        self.ws_pairs = {sym: self._ws_pair(sym) for sym in self.symbols}
        self.pair_aliases = self._build_pair_aliases()

    def _build_rest_clients(self) -> Dict[str, KrakenLiveClient]:
        out = {}
        for sym in self.symbols:
            client = KrakenLiveClient(self.cfg_loader)
            client.pair = self.metadata.get(sym, {}).get("kraken_pair", sym)
            out[sym] = client
        return out

    def _ws_pair(self, sym: str) -> str:
        meta = self.metadata.get(sym, {})
        pair = meta.get("symbol") or meta.get("kraken_pair") or sym
        if "/" in pair:
            base, quote = pair.split("/", 1)
            if meta.get("kraken_pair", "").startswith("XBT"):
                base = "XBT"
            return f"{base}/{quote}"
        kp = meta.get("kraken_pair", sym)
        quote = meta.get("quote", "")
        base = meta.get("base", kp.replace(quote, ""))
        if kp.startswith("XBT"):
            base = "XBT"
        return f"{base}/{quote}" if quote else kp

    def _build_pair_aliases(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for sym in self.symbols:
            meta = self.metadata.get(sym, {})
            variants = {
                sym,
                meta.get("kraken_pair", ""),
                meta.get("symbol", ""),
                self._ws_pair(sym),
            }
            for variant in variants:
                if variant:
                    mapping[variant.replace("/", "").upper()] = sym
                    mapping[variant.upper()] = sym
        return mapping

    def _resolve_asset(self, pair: str) -> Optional[str]:
        if not pair:
            return None
        return self.pair_aliases.get(pair.upper()) or self.pair_aliases.get(pair.replace("/", "").upper())

    async def run(self):
        while True:
            try:
                await self._connect_and_stream()
            except Exception as exc:
                LOG.error("[MarketData] WS stream error: %s. Falling back to REST polling.", exc)
                await self._run_rest_fallback(cycles=5)

    async def _connect_and_stream(self):
        LOG.info("[MarketData] Connecting to Kraken WebSocket...")
        async with websockets.connect(self.ws_url, max_queue=None) as ws:
            await self._subscribe(ws)
            LOG.info("[MarketData] Subscribed. Awaiting data...")
            async for msg in ws:
                self._handle_message(msg)

    async def _subscribe(self, ws):
        pairs = [self.ws_pairs[sym] for sym in self.symbols]
        req = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ohlc", "interval": 1},
        }
        LOG.info("[MarketData] Subscribing: %s", pairs)
        await ws.send(json.dumps(req))

    async def _run_rest_fallback(self, cycles: int = 5):
        sleep = min(float(self.cfg.get("live", {}).get("update_interval_sec", 60)), 5.0)
        for _ in range(max(cycles, 1)):
            for asset, client in self.rest_clients.items():
                try:
                    candle = client.poll_latest()
                    if candle is None:
                        continue
                    candle["asset"] = asset
                    candle["dt"] = datetime.utcfromtimestamp(int(candle["timestamp"]))
                    self._on_closed_1m(asset, candle)
                except Exception as exc:
                    LOG.error("[MarketData] REST fallback error for %s: %s", asset, exc)
            await asyncio.sleep(sleep)

    def _handle_message(self, msg: str):
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return

        if isinstance(data, dict):
            return
        if len(data) < 4 or not isinstance(data[1], list):
            return

        ohlc = data[1]
        pair = data[3]
        if len(ohlc) < 8:
            return

        asset = self._resolve_asset(pair)
        if not asset:
            return

        candle = self._parse_ws_candle(ohlc, asset)
        current = self.running_1m.get(asset)
        if current is None:
            self.running_1m[asset] = candle
            return

        if int(candle["timestamp"]) == int(current["timestamp"]):
            self.running_1m[asset] = candle
            return

        if int(candle["timestamp"]) > int(current["timestamp"]):
            self._on_closed_1m(asset, current)
            self.running_1m[asset] = candle

    @staticmethod
    def _parse_ws_candle(ohlc, asset: str) -> Dict[str, Any]:
        end_ts = int(float(ohlc[1]))
        return {
            "dt": datetime.utcfromtimestamp(end_ts),
            "timestamp": end_ts,
            "open": float(ohlc[2]),
            "high": float(ohlc[3]),
            "low": float(ohlc[4]),
            "close": float(ohlc[5]),
            "volume": float(ohlc[7]),
            "asset": asset,
        }

    def _on_closed_1m(self, asset: str, candle: Dict[str, Any]):
        self.state[asset].push_1m(candle)
        if self.dashboard:
            self.dashboard.update_candles({asset: candle})

        emits = self.tf[asset].push_1m(candle)
        for tf, bar in emits.items():
            self.state[asset].push_tf(tf, bar)
            if tf == "15m" and self.forward:
                enriched = self.enricher.enrich(self.state[asset], asset, bar)
                if enriched is None and self.strict_gate_mode:
                    LOG.info("[MarketData] strict_mode drop %s 15m bar due to missing HTF context", asset)
                else:
                    self.forward.on_bar(asset, enriched or bar)
            if self.dashboard and hasattr(self.dashboard, "update_tf_bar"):
                self.dashboard.update_tf_bar(asset, tf, bar)
            elif self.dashboard:
                self.dashboard.log_event(
                    "tf_bar",
                    None,
                    {"asset": asset, "timeframe": tf, **bar},
                )

    def latest_snapshot(self, asset: str) -> Dict[str, Any]:
        return self.state[asset].snapshot()
