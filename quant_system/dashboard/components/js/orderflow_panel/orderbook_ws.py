"""
orderbook_ws.py

Kraken L2 orderbook subscription + message normalization.
"""

import json
import asyncio
import websockets
import traceback
from typing import Callable, List

from quant_system.utils.logger import get_logger

LOG = get_logger("orderbook_ws")


class KrakenOrderbookFeed:

    def __init__(self, symbol: str, depth: int = 25):
        self.symbol = symbol
        self.depth = depth
        self.on_orderbook: Callable[[List, List], None] = lambda b, a: None
        self.ws_url = "wss://ws.kraken.com"

    # -------------------------------------------------------
    async def connect(self):
        LOG.info(f"Connecting Kraken L2 feed: {self.symbol} depth={self.depth}")
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [self.symbol],
                "subscription": {"name": "book", "depth": self.depth}
            }))

            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if isinstance(data, dict) and data.get("event"):
                        continue

                    book_data = data[1]
                    bids = book_data.get("b", [])
                    asks = book_data.get("a", [])

                    norm_bids = [[float(p), float(s)] for p, s in bids]
                    norm_asks = [[float(p), float(s)] for p, s in asks]

                    self.on_orderbook(norm_bids, norm_asks)

                except Exception as e:
                    LOG.error(f"Orderbook parse error: {e}")
                    LOG.error(traceback.format_exc())

    # -------------------------------------------------------
    def run(self):
        asyncio.get_event_loop().run_until_complete(self.connect())
