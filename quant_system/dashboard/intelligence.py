from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from quant_system.dashboard.data_access import DashboardContext, normalize_trade_frame
from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.gates import GateEvaluator


STRUCTURE_PATTERNS = ("swing", "bos", "choch", "zone", "fvg", "sweep", "retest", "bias", "pd_")
LIQUIDITY_PATTERNS = ("liq", "liquidity", "wick", "pool", "dist", "eql")
MOMENTUM_PATTERNS = ("flow", "momo", "ema", "slope", "vwap", "volume_z", "displacement", "momentum")
REGIME_PATTERNS = ("regime", "toxicity", "compression", "trend_", "expansion", "collapse", "persist")
RISK_PATTERNS = ("hazard", "evr", "median_r", "stop", "cvar", "tail", "drawdown", "risk", "edp", "eop")


def _to_row_dict(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[-1]
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def latest_market_frame(context: DashboardContext) -> pd.DataFrame:
    forward = context.forward.get("candles", pd.DataFrame())
    if isinstance(forward, pd.DataFrame) and not forward.empty:
        return forward.copy()
    candles = context.backtest.get("candles", pd.DataFrame())
    if isinstance(candles, pd.DataFrame):
        return candles.copy()
    return pd.DataFrame()


def latest_market_row(context: DashboardContext) -> Dict[str, Any]:
    return _to_row_dict(latest_market_frame(context))


def latest_reasoning(context: DashboardContext) -> Dict[str, Any]:
    events = list(context.forward.get("events", []) or [])
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        reasoning = payload.get("reasoning")
        if reasoning:
            return reasoning
        if payload:
            return payload
    return {}


def grouped_features(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    groups = {
        "structure": {},
        "liquidity": {},
        "momentum": {},
        "regime": {},
        "risk": {},
    }
    for key, value in row.items():
        if value is None:
            continue
        if any(token in str(key) for token in STRUCTURE_PATTERNS):
            groups["structure"][key] = value
        if any(token in str(key) for token in LIQUIDITY_PATTERNS):
            groups["liquidity"][key] = value
        if any(token in str(key) for token in MOMENTUM_PATTERNS):
            groups["momentum"][key] = value
        if any(token in str(key) for token in REGIME_PATTERNS):
            groups["regime"][key] = value
        if any(token in str(key) for token in RISK_PATTERNS):
            groups["risk"][key] = value
    return groups


def recent_regime_history(context: DashboardContext, limit: int = 240) -> pd.DataFrame:
    frame = latest_market_frame(context)
    if frame.empty:
        return pd.DataFrame()
    regime_cols = [c for c in frame.columns if c.startswith("p_regime_")]
    if not regime_cols:
        return pd.DataFrame()
    ts_col = "timestamp" if "timestamp" in frame.columns else "dt"
    history = frame[[ts_col] + regime_cols + ([c for c in ["regime_state", "zone_score_6h", "compression_12h", "toxicity_12h"] if c in frame.columns])].copy()
    history = history.tail(limit)
    history["timestamp"] = pd.to_datetime(history[ts_col], errors="coerce")
    return history.drop(columns=[c for c in [ts_col] if c != "timestamp" and c in history.columns])


def execution_snapshot(context: DashboardContext) -> Dict[str, Any]:
    row = latest_market_row(context)
    if not row:
        return {}

    side = row.get("side")
    if side not in {"long", "short"}:
        side = "long" if float(row.get("p_regime_trend", 0.0) or 0.0) >= float(row.get("p_regime_collapse", 0.0) or 0.0) else "short"

    gates = GateEvaluator(context.config).evaluate(row, side)
    confluence = ConfluenceEngine(context.config).evaluate(pd.Series(row))
    return {
        "side": side,
        "gates": gates,
        "confluence": confluence,
        "evr": row.get("evr"),
        "median_r": row.get("median_r"),
        "hazard": row.get("hazard_score", row.get("hazard")),
    }


def candidate_frame(context: DashboardContext) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for event in context.forward.get("events", []):
        if not isinstance(event, dict):
            continue
        payload = event.get("payload", {}) or {}
        reasoning = payload.get("reasoning", {})
        regime = reasoning.get("regime", {}) if isinstance(reasoning, dict) else {}
        flow = reasoning.get("flow", {}) if isinstance(reasoning, dict) else {}
        final = reasoning.get("final_decision", {}) if isinstance(reasoning, dict) else {}
        if not payload and not reasoning:
            continue
        rows.append(
            {
                "source": "forward_event",
                "timestamp": event.get("timestamp"),
                "trade_id": event.get("trade_id"),
                "asset": payload.get("asset") or payload.get("symbol"),
                "tier": payload.get("tier") or final.get("tier"),
                "confluence": payload.get("confluence") or final.get("confluence"),
                "evr": (payload.get("evr") or final.get("evr") or {}).get("evr") if isinstance(payload.get("evr") or final.get("evr"), dict) else payload.get("evr") or final.get("evr"),
                "median_r": (payload.get("evr") or final.get("evr") or {}).get("median_r") if isinstance(payload.get("evr") or final.get("evr"), dict) else final.get("median_r"),
                "flow_1h": flow.get("p_flow_1h") or payload.get("flow_1h"),
                "hazard": (reasoning.get("hazard", {}) if isinstance(reasoning, dict) else {}).get("hazard_score"),
                "regime_state": regime.get("regime_state"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
    else:
        trades = normalize_trade_frame(context.backtest.get("trades", pd.DataFrame())).copy()
        if trades.empty:
            return pd.DataFrame()
        df = trades.rename(columns={"entry_ts": "timestamp", "conf": "confluence", "r": "median_r"})
        df["source"] = "backtest_trade"
        df["flow_1h"] = None
        df["hazard"] = df.get("hazard_entry")
        df["regime_state"] = df.get("regime")
        df = df[["source", "timestamp", "trade_id", "asset", "tier", "confluence", "evr", "median_r", "flow_1h", "hazard", "regime_state"]]

    for col in ["confluence", "evr", "median_r", "flow_1h", "hazard"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["signal_score"] = (
        df["confluence"].fillna(0.0).clip(lower=0.0)
        * (1.0 + df["evr"].fillna(0.0).clip(lower=0.0))
        * (1.0 + df["median_r"].fillna(0.0).clip(lower=0.0))
        * (1.0 + df["flow_1h"].fillna(0.0).clip(lower=0.0))
        * (1.0 - df["hazard"].fillna(0.0).clip(lower=0.0, upper=0.95))
    )
    return df.sort_values("signal_score", ascending=False).reset_index(drop=True)


def risk_surface(context: DashboardContext) -> pd.DataFrame:
    state = context.forward.get("state", {}) or {}
    row = latest_market_row(context)
    snapshot = execution_snapshot(context)
    metrics: List[Tuple[str, float, str]] = [
        ("hazard", float(snapshot.get("hazard") or 0.0), "Lower is better"),
        ("toxicity_12h", float(row.get("toxicity_12h") or 0.0), "Macro whip risk"),
        ("cooling_active", 1.0 if state.get("cooling_to") else 0.0, "Capital lockout"),
        ("drawdown_proxy", float(state.get("max_drawdown") or 0.0), "System stress"),
        ("wick_pressure", float(row.get("liq_wick_pressure") or row.get("wick_pressure") or 0.0), "Microstructure stress"),
        ("liq_distance", float(row.get("liq_near_pool_dist") or 0.0), "Distance to pool"),
        ("volatility", float(row.get("vol_zscore") or row.get("atr_pct") or 0.0), "Current expansion"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "description"])


def operator_summary(context: DashboardContext) -> Dict[str, Any]:
    backtest = context.backtest.get("summary", {}) or {}
    forward = context.forward.get("state", {}) or {}
    latest = latest_market_row(context)
    return {
        "backtest_equity": backtest.get("ending_equity"),
        "live_equity": forward.get("equity"),
        "locked_profit": forward.get("locked_profit"),
        "open_trades": len(forward.get("open_trades", {})),
        "regime_state": latest.get("regime_state"),
        "flow_1h": latest.get("p_flow_1h", latest.get("prob_flow_1h")),
        "confluence": latest.get("confluence_score", latest.get("confluence")),
        "hazard": latest.get("hazard_score", latest.get("hazard")),
    }
