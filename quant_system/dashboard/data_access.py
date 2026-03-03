from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import pandas as pd

from quant_system.config.config_loader import ConfigLoader


CANONICAL_TRADE_COLUMNS = [
    "trade_id",
    "asset",
    "side",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "pnl",
    "r",
    "tier",
    "conf",
    "evr",
    "risk",
    "override",
    "reason",
    "stop_price",
    "size_usd",
    "qty",
    "leg",
    "regime",
    "session",
    "hazard_entry",
    "gates",
    "gate_reasons",
    "result",
]

DEFAULT_TELEMETRY_URL = os.environ.get("QUANT_TERMINAL_API_BASE", "").strip()


def _safe_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def _safe_csv(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(parse_dates or []))


def _safe_http_json(url: str, timeout: float = 1.5) -> Any:
    try:
        req = urllib_request.Request(url, headers={"Accept": "application/json"})
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, ValueError, OSError):
        return None


def _coerce_frame(data: Any, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict) and "records" in data:
        frame = pd.DataFrame(list(data.get("records") or []))
        columns = data.get("columns") or []
        if columns and frame.empty:
            frame = pd.DataFrame(columns=list(columns))
        elif columns:
            for col in columns:
                if col not in frame.columns:
                    frame[col] = None
            frame = frame[list(columns)]
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, (list, tuple)):
        frame = pd.DataFrame(list(data))
    else:
        frame = pd.DataFrame()

    for col in list(parse_dates or []):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [_jsonable(rec) for rec in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(k): _jsonable(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()}
    try:
        return float(value)
    except Exception:
        return str(value)


def frame_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    frame = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return {
        "columns": list(frame.columns),
        "records": _jsonable(frame),
    }


def _discover_first(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def normalize_trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_TRADE_COLUMNS)

    out = df.copy()
    rename_map = {
        "entry_time": "entry_ts",
        "exit_time": "exit_ts",
        "realized_r": "r",
        "r_mult": "r",
        "entry": "entry_price",
        "exit": "exit_price",
        "stop": "stop_price",
        "hazard_at_entry": "hazard_entry",
        "regime_state": "regime",
        "timestamp": "entry_ts",
        "size": "size_usd",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    for col in ("entry_ts", "exit_ts"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = pd.NaT

    numeric_defaults = {
        "entry_price": 0.0,
        "exit_price": 0.0,
        "pnl": 0.0,
        "r": 0.0,
        "conf": 0.0,
        "evr": 0.0,
        "risk": 0.0,
        "stop_price": 0.0,
        "size_usd": 0.0,
        "qty": 0.0,
        "hazard_entry": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in out.columns:
            out[col] = default

    text_defaults = {
        "trade_id": "",
        "asset": "",
        "side": "long",
        "tier": "unranked",
        "override": "",
        "reason": "",
        "leg": "core",
        "regime": "unknown",
        "session": "unknown",
    }
    for col, default in text_defaults.items():
        if col not in out.columns:
            out[col] = default

    if "result" not in out.columns:
        out["result"] = out["pnl"].apply(
            lambda value: "win" if value > 0 else ("loss" if value < 0 else "flat")
        )

    if "gates" not in out.columns:
        out["gates"] = [{} for _ in range(len(out))]
    if "gate_reasons" not in out.columns:
        out["gate_reasons"] = [[] for _ in range(len(out))]

    for col in CANONICAL_TRADE_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out.sort_values(["entry_ts", "trade_id"], na_position="last").reset_index(drop=True)
    return out[CANONICAL_TRADE_COLUMNS]


def build_equity_curve(trades: pd.DataFrame, starting_equity: float = 20_000.0) -> pd.DataFrame:
    trades = normalize_trade_frame(trades)
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])

    curve = trades[["entry_ts", "pnl"]].copy()
    curve = curve.sort_values("entry_ts")
    curve["equity"] = float(starting_equity) + curve["pnl"].fillna(0.0).cumsum()
    curve["drawdown"] = curve["equity"] - curve["equity"].cummax()
    curve = curve.rename(columns={"entry_ts": "timestamp"})
    return curve[["timestamp", "equity", "drawdown"]]


def summarize_trades(trades: pd.DataFrame, starting_equity: float = 20_000.0) -> Dict[str, Any]:
    trades = normalize_trade_frame(trades)
    equity_curve = build_equity_curve(trades, starting_equity=starting_equity)
    ending_equity = (
        float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(starting_equity)
    )
    return {
        "starting_equity": float(starting_equity),
        "ending_equity": ending_equity,
        "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
        "avg_r": float(trades["r"].mean()) if not trades.empty else 0.0,
        "max_drawdown": float(equity_curve["drawdown"].min()) if not equity_curve.empty else 0.0,
    }


def _group_reports(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    trades = normalize_trade_frame(trades)
    if trades.empty:
        return {"daily": pd.DataFrame(), "monthly": pd.DataFrame()}

    daily = trades.assign(date=trades["entry_ts"].dt.date).groupby("date").agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_r=("r", "mean"),
    ).reset_index()

    monthly = trades.assign(month=trades["entry_ts"].dt.to_period("M").astype(str)).groupby("month").agg(
        pnl=("pnl", "sum"),
        trades=("trade_id", "count"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_r=("r", "mean"),
    ).reset_index()
    return {"daily": daily, "monthly": monthly}


def serialize_backtest_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "root": str(bundle.get("root")) if bundle.get("root") is not None else None,
        "summary": _jsonable(bundle.get("summary", {})),
        "trades": frame_to_payload(normalize_trade_frame(bundle.get("trades", pd.DataFrame()))),
        "equity_curve": frame_to_payload(_coerce_frame(bundle.get("equity_curve"), parse_dates=["timestamp"])),
        "execution_log": frame_to_payload(_coerce_frame(bundle.get("execution_log"), parse_dates=["timestamp", "entry_ts", "exit_ts"])),
        "candles": frame_to_payload(_coerce_frame(bundle.get("candles"), parse_dates=["timestamp", "dt"])),
        "smc_features": frame_to_payload(_coerce_frame(bundle.get("smc_features"), parse_dates=["timestamp", "dt"])),
        "daily": frame_to_payload(_coerce_frame(bundle.get("daily"))),
        "monthly": frame_to_payload(_coerce_frame(bundle.get("monthly"))),
        "reasoning": _jsonable(bundle.get("reasoning", {})),
    }


def deserialize_backtest_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return load_backtest_bundle()
    trades = normalize_trade_frame(_coerce_frame(payload.get("trades"), parse_dates=["entry_ts", "exit_ts"]))
    summary = dict(payload.get("summary", {}) or {})
    if not summary:
        summary = summarize_trades(trades)
    return {
        "root": Path(payload["root"]) if payload.get("root") else None,
        "summary": summary,
        "trades": trades,
        "equity_curve": _coerce_frame(payload.get("equity_curve"), parse_dates=["timestamp"]),
        "execution_log": _coerce_frame(payload.get("execution_log"), parse_dates=["timestamp", "entry_ts", "exit_ts"]),
        "candles": _coerce_frame(payload.get("candles"), parse_dates=["timestamp", "dt"]),
        "smc_features": _coerce_frame(payload.get("smc_features"), parse_dates=["timestamp", "dt"]),
        "daily": _coerce_frame(payload.get("daily")),
        "monthly": _coerce_frame(payload.get("monthly")),
        "reasoning": dict(payload.get("reasoning", {}) or {}),
    }


def serialize_forward_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "root": str(bundle.get("root")) if bundle.get("root") is not None else None,
        "state": _jsonable(bundle.get("state", {})),
        "events": _jsonable(bundle.get("events", [])),
        "candles": frame_to_payload(_coerce_frame(bundle.get("candles"), parse_dates=["timestamp", "dt"])),
        "tf_bars": {
            str(tf): frame_to_payload(_coerce_frame(frame, parse_dates=["timestamp", "dt"]))
            for tf, frame in dict(bundle.get("tf_bars", {}) or {}).items()
        },
    }


def deserialize_forward_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return load_forward_bundle()
    return {
        "root": Path(payload["root"]) if payload.get("root") else None,
        "state": dict(payload.get("state", {}) or {}),
        "events": list(payload.get("events", []) or []),
        "candles": _coerce_frame(payload.get("candles"), parse_dates=["timestamp", "dt"]),
        "tf_bars": {
            str(tf): _coerce_frame(frame, parse_dates=["timestamp", "dt"])
            for tf, frame in dict(payload.get("tf_bars", {}) or {}).items()
        },
    }


def serialize_model_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return frame_to_payload(df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def deserialize_model_summary(payload: Dict[str, Any]) -> pd.DataFrame:
    return _coerce_frame(payload)


def load_backtest_bundle(base_dir: Optional[Path] = None, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if snapshot is not None and isinstance(snapshot, dict) and snapshot.get("backtest"):
        bundle = dict(snapshot.get("backtest") or {})
        trades = normalize_trade_frame(_coerce_frame(bundle.get("trades"), parse_dates=["entry_ts", "exit_ts"]))
        summary = dict(bundle.get("summary", {}) or {})
        if not summary:
            summary = summarize_trades(trades)
        return {
            "root": None,
            "summary": summary,
            "trades": trades,
            "equity_curve": _coerce_frame(bundle.get("equity_curve"), parse_dates=["timestamp"]),
            "execution_log": _coerce_frame(bundle.get("execution_log"), parse_dates=["timestamp", "entry_ts", "exit_ts"]),
            "candles": _coerce_frame(bundle.get("candles"), parse_dates=["timestamp", "dt"]),
            "smc_features": _coerce_frame(bundle.get("smc_features"), parse_dates=["timestamp", "dt"]),
            "daily": _coerce_frame(bundle.get("daily")),
            "monthly": _coerce_frame(bundle.get("monthly")),
            "reasoning": dict(bundle.get("reasoning", {}) or {}),
        }

    root = Path(base_dir) if base_dir is not None else _discover_first(
        Path.cwd() / "backtest_outputs",
        Path.cwd() / "backtest_output",
    )
    summary = _safe_json(root / "summary.json") or {}
    ledger = normalize_trade_frame(
        _safe_csv(root / "ledger.csv", parse_dates=["entry_ts", "exit_ts"])
        if (root / "ledger.csv").exists()
        else _safe_csv(root / "trades.csv", parse_dates=["entry_time", "exit_time"])
    )
    equity_curve = _safe_csv(root / "equity_curve.csv", parse_dates=["timestamp"])
    if equity_curve.empty:
        equity_curve = build_equity_curve(
            ledger,
            starting_equity=float(summary.get("starting_equity", 20_000.0) or 20_000.0),
        )
    execution_log = _safe_csv(root / "execution_log.csv", parse_dates=["timestamp", "entry_ts", "exit_ts"])
    candles = _safe_csv(root / "candles_15m.csv", parse_dates=["timestamp", "dt"])
    if candles.empty:
        candles = _safe_csv(root / "candles.csv", parse_dates=["timestamp", "dt"])
    smc = _safe_csv(root / "smc_features.csv", parse_dates=["timestamp", "dt"])
    daily = _safe_csv(root / "daily_report.csv")
    monthly = _safe_csv(root / "monthly_report.csv")
    reasoning = _safe_json(root / "reasoning.json") or {}

    if daily.empty or monthly.empty:
        grouped = _group_reports(ledger)
        daily = grouped["daily"]
        monthly = grouped["monthly"]

    if not summary:
        summary = summarize_trades(ledger)
    else:
        merged_summary = summarize_trades(
            ledger,
            starting_equity=float(summary.get("starting_equity", 20_000.0) or 20_000.0),
        )
        merged_summary.update(summary)
        summary = merged_summary

    return {
        "root": root,
        "summary": summary,
        "trades": ledger,
        "equity_curve": equity_curve,
        "execution_log": execution_log,
        "candles": candles,
        "smc_features": smc,
        "daily": daily,
        "monthly": monthly,
        "reasoning": reasoning,
    }


def _coerce_forward_state(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(snapshot.get("state", {}) or {})
    events = list(snapshot.get("events", []) or [])
    candles = snapshot.get("candles", pd.DataFrame())
    if isinstance(candles, dict):
        candles = pd.DataFrame(candles)
    elif isinstance(candles, (list, tuple)):
        candles = pd.DataFrame(list(candles))
    if candles is None:
        candles = pd.DataFrame()

    state.setdefault("equity", 20_000.0)
    state.setdefault("free_capital", state.get("equity", 20_000.0))
    state.setdefault("locked_profit", 0.0)
    state.setdefault("open_trades", {})
    return {"state": state, "events": events, "candles": candles}


def load_forward_bundle(
    adapter: Any = None,
    base_dir: Optional[Path] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if snapshot is not None and isinstance(snapshot, dict):
        bundle = dict(snapshot)
        if bundle.get("state") or bundle.get("events") or bundle.get("candles"):
            return {
                "root": None,
                **_coerce_forward_state(
                    {
                        "state": bundle.get("state", {}),
                        "events": bundle.get("events", []),
                        "candles": bundle.get("candles", pd.DataFrame()),
                    }
                ),
                "tf_bars": {
                    str(tf): _coerce_frame(frame, parse_dates=["timestamp", "dt"])
                    for tf, frame in dict(bundle.get("tf_bars", {}) or {}).items()
                },
            }

    if adapter is not None:
        try:
            snapshot = _coerce_forward_state(adapter.get_snapshot())
            if snapshot["events"] or not snapshot["candles"].empty:
                return {"root": None, **snapshot, "tf_bars": dict(adapter.get_snapshot().get("tf_bars", {}) or {})}
        except Exception:
            pass

    root = Path(base_dir) if base_dir is not None else _discover_first(
        Path.cwd() / "forward_outputs",
        Path.cwd() / "live_outputs",
    )
    raw_snapshot = _safe_json(root / "snapshot.json") or {}
    if isinstance(raw_snapshot, dict) and any(k in raw_snapshot for k in ("state", "events", "candles")):
        state = raw_snapshot.get("state", {}) or {}
        events = raw_snapshot.get("events", []) or []
        candles = raw_snapshot.get("candles", pd.DataFrame())
        return {
            "root": root,
            **_coerce_forward_state({"state": state, "events": events, "candles": candles}),
            "tf_bars": {
                str(tf): _coerce_frame(frame, parse_dates=["timestamp", "dt"])
                for tf, frame in dict(raw_snapshot.get("tf_bars", {}) or {}).items()
            },
        }

    state = raw_snapshot or _safe_json(root / "state.json") or {}
    events = _safe_json(root / "events.json") or []
    if not events and (root / "events.csv").exists():
        events_df = _safe_csv(root / "events.csv", parse_dates=["timestamp"])
        events = events_df.to_dict(orient="records")
    candles = _safe_csv(root / "candles.csv", parse_dates=["timestamp", "dt"])
    if candles.empty:
        candles = _safe_csv(root / "bars.csv", parse_dates=["timestamp", "dt"])
    return {
        "root": root,
        **_coerce_forward_state({"state": state, "events": events, "candles": candles}),
        "tf_bars": {},
    }


def load_model_registry_summary(base_dir: Optional[Path]) -> pd.DataFrame:
    if base_dir is None:
        return pd.DataFrame()
    root = Path(base_dir)
    if not root.exists():
        return pd.DataFrame()

    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        versions = sorted(p for p in model_dir.iterdir() if p.is_dir())
        if not versions:
            continue
        latest = versions[-1]
        cfg = _safe_json(latest / "config.json") or {}
        metrics = _safe_json(latest / "metrics.json") or {}
        feature_list = cfg.get("feature_names") or cfg.get("features") or []
        rows.append(
            {
                "model": model_dir.name,
                "version": latest.name,
                "model_type": cfg.get("model_type") or cfg.get("booster") or cfg.get("type"),
                "timeframe": cfg.get("timeframe") or cfg.get("label_tf"),
                "label": cfg.get("label_name") or cfg.get("target"),
                "feature_count": len(feature_list) if isinstance(feature_list, list) else None,
                "auc": metrics.get("auc"),
                "pr_auc": metrics.get("pr_auc"),
                "brier": metrics.get("brier"),
                "logloss": metrics.get("logloss"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True) if rows else pd.DataFrame()


@dataclass
class DashboardContext:
    config: Dict[str, Any]
    theme_choice: str
    model_version: str
    transport: str
    backtest_dir: Path
    forward_dir: Path
    model_dir: Path
    backtest: Dict[str, Any]
    forward: Dict[str, Any]
    model_summary: pd.DataFrame


def build_context(
    theme_choice: str,
    *,
    backtest_dir: Optional[str] = None,
    forward_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    adapter: Any = None,
    telemetry_url: Optional[str] = None,
) -> DashboardContext:
    cfg = ConfigLoader("quant_system/config").load()
    model_root = Path(
        model_dir
        or cfg.get("paths", {}).get("model_registry")
        or cfg.get("models", {}).get("registry_path")
        or "models"
    )
    bt_root = Path(backtest_dir) if backtest_dir else _discover_first(
        Path.cwd() / "backtest_outputs",
        Path.cwd() / "backtest_output",
    )
    fwd_root = Path(forward_dir) if forward_dir else _discover_first(
        Path.cwd() / "forward_outputs",
        Path.cwd() / "live_outputs",
    )
    telemetry_base = (telemetry_url if telemetry_url is not None else DEFAULT_TELEMETRY_URL).strip()
    if telemetry_base:
        query = urllib_parse.urlencode(
            {
                "backtest_dir": str(bt_root),
                "forward_dir": str(fwd_root),
                "model_dir": str(model_root),
            }
        )
        payload = _safe_http_json(f"{telemetry_base.rstrip('/')}/dashboard/context?{query}")
        if isinstance(payload, dict):
            model_summary = deserialize_model_summary(payload.get("model_summary", {}))
            model_version = str(payload.get("model_version") or (model_summary["version"].max() if not model_summary.empty else "unavailable"))
            return DashboardContext(
                config=cfg,
                theme_choice=theme_choice,
                model_version=model_version,
                transport="telemetry_api",
                backtest_dir=Path(payload.get("backtest_dir") or bt_root),
                forward_dir=Path(payload.get("forward_dir") or fwd_root),
                model_dir=Path(payload.get("model_dir") or model_root),
                backtest=deserialize_backtest_bundle(payload.get("backtest", {})),
                forward=deserialize_forward_bundle(payload.get("forward", {})),
                model_summary=model_summary,
            )

    model_summary = load_model_registry_summary(model_root)
    model_version = model_summary["version"].max() if not model_summary.empty else "unavailable"
    return DashboardContext(
        config=cfg,
        theme_choice=theme_choice,
        model_version=str(model_version),
        transport="artifacts",
        backtest_dir=bt_root,
        forward_dir=fwd_root,
        model_dir=model_root,
        backtest=load_backtest_bundle(bt_root),
        forward=load_forward_bundle(adapter=adapter, base_dir=fwd_root),
        model_summary=model_summary,
    )
