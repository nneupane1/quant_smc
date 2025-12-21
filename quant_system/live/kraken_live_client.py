"""
KrakenLiveClient:
Pulls real-time 1m OHLCV candles for BTC/USD.
Uses REST polling (safe + deterministic).
Reads API keys from .env.
"""

import os
import time
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from quant_system.data.ingest.api_retry import retry
from quant_system.utils.logger import log


class KrakenLiveClient:
    def __init__(self, config: Dict[str, Any]):
        load_dotenv()

        self.base_url = "https://api.kraken.com/0/public/OHLC"
        self.pair = "XBTUSD"   # BTCUSD is XBTUSD on Kraken

        self.interval = 1      # 1-minute candles only
        self.last_ts = None

        self.sleep = float(config["live"]["poll_seconds"])
        log("KrakenLiveClient initialized for BTC/USD (XBTUSD).")

    # ---------------------------------------------------------------
    @api_retry
    def _fetch_ohlc(self) -> Dict[str, Any]:
        params = {"pair": self.pair, "interval": self.interval}
        r = requests.get(self.base_url, params=params)
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
            log(f"Kraken returned error: {data['error']}")
            return None

        candles = data["result"][self.pair]
        if len(candles) == 0:
            return None

        latest = self._parse_candle(candles[-1])

        if self.last_ts is None:
            self.last_ts = latest["timestamp"]
            log(f"Initial candle received ts={self.last_ts}")
            return latest

        if latest["timestamp"] > self.last_ts:
            self.last_ts = latest["timestamp"]
            log(f"New 1m candle ts={self.last_ts} close={latest['close']}")
            return latest

        # no new candle yet
        return None

    # ---------------------------------------------------------------
    def run_stream(self):
        """
        Infinite generator that yields new closed 1m candles.
        """
        log("Starting BTC/USD live stream.")
        while True:
            try:
                c = self.poll_latest()
                if c is not None:
                    yield c
                time.sleep(self.sleep)
            except Exception as e:
                log(f"Stream error: {e}")
                time.sleep(self.sleep)
