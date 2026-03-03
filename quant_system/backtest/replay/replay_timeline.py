"""
Replay timeline helpers.

Transforms either:
 - a trade ledger produced by TradeLog/Backtester, or
 - an existing event log

into a normalized event timeline usable by ReplayController.
"""

from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _as_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, dict)):
        return pd.DataFrame(list(data))
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def _normalize_timestamp(df: pd.DataFrame, candles: Optional[pd.DataFrame]) -> pd.DataFrame:
    ts_col = None
    for cand in ("timestamp", "dt", "entry_ts", "exit_ts"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    if candles is not None and not candles.empty:
        candle_ts_col = "timestamp" if "timestamp" in candles.columns else "dt"
        candle_times = pd.to_datetime(candles[candle_ts_col], errors="coerce").reset_index(drop=True)
        mapped_idx = []
        for ts in df["timestamp"]:
            if pd.isna(ts):
                mapped_idx.append(None)
                continue
            matches = candle_times[candle_times == ts]
            mapped_idx.append(int(matches.index[0]) if not matches.empty else None)
        df["candle_idx"] = mapped_idx
    return df


def build_timeline(execution_log: Any, candles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Returns normalized replay events with columns like:
      timestamp, candle_idx, trade_id, type, side, price, reason, asset, leg
    """
    df = _as_frame(execution_log)
    if df.empty:
        return df

    if {"timestamp", "type"}.issubset(df.columns):
        out = _normalize_timestamp(df, candles)
        sort_cols = [c for c in ["timestamp", "trade_id", "type"] if c in out.columns]
        return out.sort_values(sort_cols, na_position="last").reset_index(drop=True) if sort_cols else out.reset_index(drop=True)

    # Trade ledger -> event stream
    if "entry_ts" in df.columns:
        events = []
        for _, row in df.iterrows():
            base: Dict[str, Any] = {
                "trade_id": row.get("trade_id"),
                "asset": row.get("asset"),
                "side": row.get("side"),
                "leg": row.get("leg"),
                "tier": row.get("tier"),
            }
            events.append(
                {
                    **base,
                    "timestamp": row.get("entry_ts"),
                    "type": "entry",
                    "price": row.get("entry_price"),
                    "reason": "entry",
                }
            )
            if pd.notna(row.get("exit_ts")):
                reason = row.get("reason", "exit")
                event_type = "stop" if "stop" in str(reason).lower() else "exit"
                events.append(
                    {
                        **base,
                        "timestamp": row.get("exit_ts"),
                        "type": event_type,
                        "price": row.get("exit_price"),
                        "reason": reason,
                        "pnl": row.get("pnl"),
                        "r": row.get("r"),
                    }
                )
        out = pd.DataFrame(events)
        out = _normalize_timestamp(out, candles)
        sort_cols = [c for c in ["timestamp", "trade_id", "type"] if c in out.columns]
        return out.sort_values(sort_cols, na_position="last").reset_index(drop=True) if sort_cols else out.reset_index(drop=True)

    out = _normalize_timestamp(df, candles)
    return out.sort_values("timestamp", na_position="last").reset_index(drop=True) if "timestamp" in out.columns else out.reset_index(drop=True)
