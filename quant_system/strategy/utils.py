from typing import Iterable, Optional, Tuple


def tokens_for_risk(equity: float, risk_pct: float, entry: float, stop: float, side: str) -> float:
    """Risk % of equity between entry and stop. Returns quantity (base asset)."""
    if entry <= 0 or stop <= 0 or equity <= 0 or risk_pct <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    qty = risk_amount / per_unit_risk
    return max(0.0, qty)


def r_multiple(entry: float, stop: float, last: float, side: str) -> float:
    """Current R multiple based on last price (unrealized)."""
    if entry <= 0 or stop <= 0:
        return 0.0
    risk = abs(entry - stop)
    move = (last - entry) if side == "long" else (entry - last)
    return move / risk if risk > 0 else 0.0


def throttle_by_absorption(absorption: float, lo: float, hi: float) -> float:
    """
    Returns size multiplier ∈ [0.5, 1.0]; 1.0 at ≤lo, 0.5 at ≥hi; linear in between.
    """
    if absorption <= lo:
        return 1.0
    if absorption >= hi:
        return 0.5
    t = (absorption - lo) / (hi - lo + 1e-9)
    return 1.0 - 0.5 * t


def first_number(row: dict, keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        try:
            number = float(value)
        except Exception:
            continue
        if number == number:
            return number
    return float(default)


def first_text(row: dict, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}
