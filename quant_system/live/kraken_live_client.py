"""
Kraken live market-data client with a safe paper-order fallback.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from quant_system.data.ingest.api_retry import retry
from quant_system.utils.logger import get_logger

LOG = get_logger("kraken_live_client")


def _as_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class KrakenLiveClient:
    def __init__(self, config: Any):
        load_dotenv()
        cfg = _as_dict(config)
        self.cfg = cfg

        self.base_url = "https://api.kraken.com/0/public/OHLC"
        self.private_base_url = "https://api.kraken.com/0/private/AddOrder"
        self.pair = (
            cfg.get("live_trading", {}).get("venue", {}).get("pair")
            or cfg.get("data", {}).get("pair")
            or "XBTUSD"
        )

        self.interval = 1
        self.last_ts = None

        self.sleep = float(cfg.get("live", {}).get("update_interval_sec", 60))
        self.live_enabled = bool(cfg.get("live_trading", {}).get("enabled", False))
        self.api_url = cfg.get("api", {}).get("kraken", {}).get("url", "https://api.kraken.com")
        self.api_key = (
            cfg.get("api", {}).get("kraken", {}).get("key")
            or os.getenv("KRAKEN_API_KEY")
        )
        self.api_secret = (
            cfg.get("api", {}).get("kraken", {}).get("secret")
            or os.getenv("KRAKEN_API_SECRET")
        )
        LOG.info("KrakenLiveClient initialized for pair=%s live_enabled=%s", self.pair, self.live_enabled)

    # ---------------------------------------------------------------
    @retry
    def _fetch_ohlc(self) -> Dict[str, Any]:
        params = {"pair": self.pair, "interval": self.interval}
        r = requests.get(self.base_url, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"Kraken error {r.status_code}: {r.text}")

        return r.json()

    # ---------------------------------------------------------------
    def _parse_candle(self, arr) -> Dict[str, Any]:
        # Kraken OHLC columns:
        # [0]=timestamp, [1]=open, [2]=high, [3]=low, [4]=close, [5]=vwap, [6]=volume, [7]=count
        return {
            "timestamp": int(arr[0]),
            "open": float(arr[1]),
            "high": float(arr[2]),
            "low": float(arr[3]),
            "close": float(arr[4]),
            "volume": float(arr[6]),
        }

    # ---------------------------------------------------------------
    def poll_latest(self) -> Optional[Dict[str, Any]]:
        data = self._fetch_ohlc()

        if "error" in data and len(data["error"]) > 0:
            LOG.warning("Kraken returned error: %s", data["error"])
            return None

        candles = data["result"].get(self.pair) or next(
            (value for key, value in data["result"].items() if key != "last"),
            [],
        )
        if len(candles) == 0:
            return None

        latest = self._parse_candle(candles[-1])
        latest["asset"] = self.pair

        if self.last_ts is None:
            self.last_ts = latest["timestamp"]
            LOG.info("Initial candle received ts=%s", self.last_ts)
            return latest

        if latest["timestamp"] > self.last_ts:
            self.last_ts = latest["timestamp"]
            LOG.info("New 1m candle ts=%s close=%s", self.last_ts, latest["close"])
            return latest

        # no new candle yet
        return None

    # ---------------------------------------------------------------
    def run_stream(self):
        """
        Infinite generator that yields new closed 1m candles.
        """
        LOG.info("Starting Kraken live stream for %s.", self.pair)
        while True:
            try:
                c = self.poll_latest()
                if c is not None:
                    yield c
                time.sleep(self.sleep)
            except Exception as e:
                LOG.error("Stream error: %s", e)
                time.sleep(self.sleep)

    def submit_order(
        self,
        *,
        pair: str,
        side: str,
        volume: float,
        price: Optional[float] = None,
        leverage: int = 1,
        ordertype: str = "market",
    ) -> Dict[str, Any]:
        """
        Submit a live order when enabled, otherwise return a deterministic paper fill.
        """
        if not self.live_enabled:
            txid = f"paper-{int(time.time() * 1000)}"
            LOG.info(
                "Paper order accepted pair=%s side=%s volume=%s price=%s leverage=%s txid=%s",
                pair,
                side,
                volume,
                price,
                leverage,
                txid,
            )
            return {"txid": txid, "paper": True}

        if not self.api_key or not self.api_secret:
            raise RuntimeError("Live trading enabled but Kraken API credentials are missing.")

        nonce = str(int(time.time() * 1000))
        payload = {
            "nonce": nonce,
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        if price is not None and ordertype != "market":
            payload["price"] = str(price)
        if leverage and leverage > 1:
            payload["leverage"] = str(leverage)

        encoded = urllib.parse.urlencode(payload)
        signature = self._sign("/0/private/AddOrder", nonce, encoded)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
        }
        response = requests.post(self.private_base_url, data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken order error: {data['error']}")
        txids = data.get("result", {}).get("txid", [])
        txid = txids[0] if isinstance(txids, list) and txids else data.get("result", {}).get("descr", {}).get("order")
        return {"txid": txid, "paper": False, "raw": data}

    def _sign(self, path: str, nonce: str, post_data: str) -> str:
        """
        Minimal Kraken API signature helper for private endpoints.
        """
        import base64

        secret = base64.b64decode(self.api_secret)
        message = path.encode() + hashlib.sha256((nonce + post_data).encode()).digest()
        return base64.b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode()
