"""
Utility functions for label generation.
Minimal docstrings as requested.
"""

from typing import List, Dict
from quant_system.data.store.datamodel import Candle


def forward_returns(closes: List[float], i: int, horizon: int) -> List[float]:
    """Compute forward returns for horizon bars."""
    res = []
    end = min(len(closes), i + horizon + 1)
    c0 = closes[i]
    if c0 <= 0:
        return [0.0]
    for j in range(i + 1, end):
        res.append((closes[j] - c0) / c0)
    return res


def forward_min_price(lows: List[float], i: int, horizon: int) -> float:
    """Minimum future low."""
    end = min(len(lows), i + horizon + 1)
    return min(lows[i+1:end]) if i + 1 < end else lows[i]


def forward_max_price(highs: List[float], i: int, horizon: int) -> float:
    """Maximum future high."""
    end = min(len(highs), i + horizon + 1)
    return max(highs[i+1:end]) if i + 1 < end else highs[i]


def price_r_move(atr: float, r_mult: float) -> float:
    """R-multiple in price units."""
    return atr * r_mult


def direction_from_sweep(sweep_up: int, sweep_down: int) -> int:
    """Convert sweep flags to direction (+1 long, -1 short)."""
    if sweep_down:
        return 1
    if sweep_up:
        return -1
    return 0


def detect_invalidation(
    direction: int,
    stop_level: float,
    highs: List[float],
    lows: List[float],
    j: int
) -> bool:
    """Check if price violates stop/invalidation level."""
    if direction == 1:
        return lows[j] <= stop_level
    else:
        return highs[j] >= stop_level


def choch_against(direction: int, choch: Dict[str, int]) -> bool:
    """Check CHOCH against current direction."""
    if direction == 1:
        return choch.get("choch_down", 0) == 1
    else:
        return choch.get("choch_up", 0) == 1


def next_index(idx: Dict[int, int], ts: int) -> int:
    """Safe lookup of timestamp index."""
    return idx.get(ts, -1)


def make_entry_dict(candle: Candle, direction: int, entry_price: float) -> Dict[str, float]:
    """Format entry dict for hazard labeler."""
    return {
        "direction": direction,
        "entry_price": entry_price
    }
