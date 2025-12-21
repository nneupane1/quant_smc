"""
Datamodels for OHLCV and Timeframe Batches
------------------------------------------

Defines strongly typed, immutable containers for use across:
- Ingestion
- Timeframe builder
- Feature engineering
- Labels
- Backtesting and forward testing

Designed for:
- Safety (type correctness)
- Performance (lightweight)
- CSV-native compatibility
"""

from dataclasses import dataclass
from typing import List, Optional


# ------------------------------------------------------------
# Base OHLCV Candle
# ------------------------------------------------------------
@dataclass(frozen=True)
class Candle:
    """
    Immutable OHLCV candle.

    Attributes:
        timestamp: int (unix seconds, UTC)
        open: float
        high: float
        low: float
        close: float
        volume: float
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_row(self) -> List:
        """Return candle as CSV-friendly row."""
        return [
            self.timestamp,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume
        ]

    @staticmethod
    def from_csv_row(row: dict) -> "Candle":
        """Convert a CSV dict row into a Candle."""
        return Candle(
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def is_valid(self) -> bool:
        """Basic sanity validation."""
        return (
            self.open > 0 and self.high > 0 and self.low > 0 and self.close > 0
            and self.high >= max(self.open, self.close)
            and self.low <= min(self.open, self.close)
        )


# ------------------------------------------------------------
# Higher-Timeframe Candle Batch
# ------------------------------------------------------------
@dataclass
class TFCandleBatch:
    """
    A batch of candles for a given timeframe.

    Attributes:
        timeframe: str (1m, 15m, 1h, 6h, 12h)
        candles: list of Candle
    """

    timeframe: str
    candles: List[Candle]

    def __len__(self) -> int:
        return len(self.candles)

    def append(self, candle: Candle) -> None:
        self.candles.append(candle)

    def extend(self, batch: "TFCandleBatch") -> None:
        self.candles.extend(batch.candles)

    def first_ts(self) -> Optional[int]:
        return self.candles[0].timestamp if self.candles else None

    def last_ts(self) -> Optional[int]:
        return self.candles[-1].timestamp if self.candles else None

    def as_rows(self) -> List[List]:
        """Return all candles as CSV rows."""
        return [c.as_row() for c in self.candles]


# ------------------------------------------------------------
# Optional tick-level model for live forward testing
# ------------------------------------------------------------
@dataclass(frozen=True)
class Tick:
    """
    Lightweight tick model for forward-test and streaming evaluation.

    Attributes:
        timestamp: int
        price: float
        volume: float
    """

    timestamp: int
    price: float
    volume: float

    def merge_with_candle(self, candle: Candle) -> Candle:
        """
        Update a Candle with tick information.
        Used in forward testing where 1m bars are constructed from ticks.
        """
        high = max(candle.high, self.price)
        low = min(candle.low, self.price)

        return Candle(
            timestamp=candle.timestamp,
            open=candle.open,
            high=high,
            low=low,
            close=self.price,
            volume=candle.volume + self.volume,
        )
