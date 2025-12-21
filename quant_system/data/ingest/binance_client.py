"""
BinanceClient
-------------
Thin wrapper around ccxt.binance to fetch historical OHLCV.
Respects start/end timestamps (seconds) and enforces 1m resolution.
"""

import time
from typing import List, Dict, Any

import ccxt


class BinanceClient:
    def __init__(self, pair: str = "BTC/USDT", rate_limit: bool = True):
        self.pair = pair
        self.client = ccxt.binance({"enableRateLimit": rate_limit})

    def fetch_ohlcv(
        self,
        start_ts: int,
        end_ts: int,
        batch_sleep: float = 0.2,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch 1m OHLCV from start_ts to end_ts (inclusive) in seconds.
        Returns list of dicts with timestamp (seconds) and ohlcv fields.
        """
        tf = "1m"
        since_ms = start_ts * 1000
        end_ms = end_ts * 1000

        out: List[Dict[str, Any]] = []

        while since_ms <= end_ms:
            ohlcv = self.client.fetch_ohlcv(self.pair, timeframe=tf, since=since_ms, limit=limit)
            if not ohlcv:
                break

            for ts, o, h, l, c, v in ohlcv:
                if ts > end_ms:
                    break
                out.append(
                    {
                        "timestamp": int(ts // 1000),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                    }
                )

            # Advance cursor to last timestamp + 1 minute
            last_ts = ohlcv[-1][0]
            since_ms = last_ts + 60_000
            time.sleep(batch_sleep)

        return out
