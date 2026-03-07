"""
Kraken live market-data client with a safe paper-order fallback.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any, Dict, Optional, Sequence

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

        self.api_url = cfg.get("api", {}).get("kraken", {}).get("url", "https://api.kraken.com")
        self.base_url = f"{self.api_url}/0/public/OHLC"
        self.private_base_url = f"{self.api_url}/0/private"
        self.pair = (
            cfg.get("live_trading", {}).get("venue", {}).get("pair")
            or cfg.get("data", {}).get("pair")
            or "XBTUSD"
        )

        self.interval = 1
        self.last_ts = None
        self._last_nonce = 0

        self.sleep = float(cfg.get("live", {}).get("update_interval_sec", 60))
        self.live_enabled = bool(cfg.get("live_trading", {}).get("enabled", False))
        self.api_key = (
            cfg.get("api", {}).get("kraken", {}).get("key")
            or os.getenv("KRAKEN_API_KEY")
        )
        self.api_secret = (
            cfg.get("api", {}).get("kraken", {}).get("secret")
            or os.getenv("KRAKEN_API_SECRET")
        )
        self.api_otp = (
            cfg.get("api", {}).get("kraken", {}).get("otp")
            or os.getenv("KRAKEN_OTP")
        )
        LOG.info(
            "KrakenLiveClient initialized for pair=%s live_enabled=%s otp_enabled=%s",
            self.pair,
            self.live_enabled,
            bool(self.api_otp),
        )

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

    def _next_nonce(self) -> str:
        now = int(time.time() * 1000)
        if now <= self._last_nonce:
            now = self._last_nonce + 1
        self._last_nonce = now
        return str(now)

    @retry
    def _private_request(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Kraken private endpoint requested without API key/secret configured.")

        nonce = self._next_nonce()
        body = dict(payload or {})
        body["nonce"] = nonce
        if self.api_otp:
            body["otp"] = self.api_otp

        encoded = urllib.parse.urlencode(body)
        path = f"/0/private/{endpoint}"
        signature = self._sign(path, nonce, encoded)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
        }

        response = requests.post(f"{self.private_base_url}/{endpoint}", data=body, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken private {endpoint} error: {data['error']}")
        return data.get("result", {})

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

        payload = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        if price is not None and ordertype != "market":
            payload["price"] = str(price)
        if leverage and leverage > 1:
            payload["leverage"] = str(leverage)

        result = self._private_request("AddOrder", payload)
        txids = result.get("txid", [])
        txid = txids[0] if isinstance(txids, list) and txids else result.get("descr", {}).get("order")
        return {"txid": txid, "paper": False, "raw": result}

    def cancel_order(self, txid: str) -> Dict[str, Any]:
        if not self.live_enabled:
            return {"paper": True, "txid": txid, "status": "canceled"}
        result = self._private_request("CancelOrder", {"txid": txid})
        return {"paper": False, "txid": txid, "raw": result}

    def open_orders(self) -> Dict[str, Any]:
        if not self.live_enabled:
            return {"paper": True, "open": {}}
        result = self._private_request("OpenOrders")
        return {"paper": False, "open": result.get("open", {}), "raw": result}

    def query_orders(self, txids: Sequence[str]) -> Dict[str, Any]:
        if not txids:
            return {"paper": (not self.live_enabled), "orders": {}}
        if not self.live_enabled:
            return {
                "paper": True,
                "orders": {txid: {"status": "closed", "descr": {"order": "paper"}} for txid in txids},
            }
        result = self._private_request("QueryOrders", {"txid": ",".join(txids)})
        return {"paper": False, "orders": result, "raw": result}

    def balance(self) -> Dict[str, Any]:
        if not self.live_enabled:
            return {"paper": True, "balances": {}}
        result = self._private_request("Balance")
        return {"paper": False, "balances": result, "raw": result}

    def _sign(self, path: str, nonce: str, post_data: str) -> str:
        """
        Minimal Kraken API signature helper for private endpoints.
        """
        secret = base64.b64decode(self.api_secret)
        message = path.encode() + hashlib.sha256((nonce + post_data).encode()).digest()
        return base64.b64encode(hmac.new(secret, message, hashlib.sha512).digest()).decode()
