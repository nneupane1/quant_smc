"""
Standalone replay exporter.

Exports a self-contained HTML file from current backtest artifacts or from a
direct replay payload. The exporter normalizes the repaired repo contracts:
 - candles from backtest/forward/live bundles
 - equity curves or raw pnl series
 - normalized trade ledgers
 - reasoning maps keyed by timestamp
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from quant_system.dashboard.data_access import (
    build_equity_curve,
    load_backtest_bundle,
    normalize_trade_frame,
)
from quant_system.utils.logger import get_logger

LOG = get_logger("replay_exporter")


class ReplayExporter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.assets_dir = Path(__file__).resolve().parent

    def export(self, replay_data: Dict[str, Any]) -> Path:
        LOG.info("Preparing replay export")
        payload = self._normalize_payload(replay_data)
        html = self._build_html(payload)
        self.output_path.write_text(html, encoding="utf-8")
        LOG.info("Replay exported successfully: %s", self.output_path)
        return self.output_path

    def export_backtest_bundle(
        self,
        *,
        base_dir: Optional[str | Path] = None,
        bundle: Optional[Dict[str, Any]] = None,
        starting_equity: float = 20_000.0,
    ) -> Path:
        source = bundle or load_backtest_bundle(Path(base_dir) if base_dir is not None else None)
        replay_data = {
            "bars": source.get("candles"),
            "pnl": source.get("equity_curve"),
            "trades": source.get("trades"),
            "reasoning": source.get("reasoning"),
            "starting_equity": starting_equity,
        }
        return self.export(replay_data)

    def _build_html(self, payload: Dict[str, Any]) -> str:
        data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        data_json = data_json.replace("</", "<\\/")
        html_template = (self.assets_dir / "template.html").read_text(encoding="utf-8")
        js = (self.assets_dir / "replay.js").read_text(encoding="utf-8")
        css = (self.assets_dir / "styles.css").read_text(encoding="utf-8")
        return (
            html_template
            .replace("/*__CSS__*/", css)
            .replace("//__JS__", js)
            .replace("__DATA__", data_json)
        )

    def _normalize_payload(self, replay_data: Dict[str, Any]) -> Dict[str, Any]:
        bar_source = replay_data.get("bars", replay_data.get("candles"))
        trade_source = replay_data.get("trades", replay_data.get("ledger"))
        pnl_source = replay_data.get("pnl", replay_data.get("equity_curve"))
        reasoning_source = replay_data.get("reasoning", replay_data.get("reasoning_map"))

        bars = self._normalize_bars(bar_source)
        trades = self._normalize_trades(trade_source, bars)
        pnl = self._normalize_pnl(
            pnl_source,
            trades,
            starting_equity=float(replay_data.get("starting_equity", 20_000.0) or 20_000.0),
        )
        reasoning = self._normalize_reasoning(reasoning_source, bars)
        return {
            "meta": {
                "bar_count": len(bars),
                "trade_count": len(trades),
                "equity_points": len(pnl),
            },
            "bars": bars,
            "pnl": pnl,
            "trades": trades,
            "reasoning": reasoning,
        }

    def _normalize_bars(self, bars: Any) -> List[Dict[str, Any]]:
        frame = self._as_frame(bars)
        if frame.empty:
            return []

        if "timestamp" not in frame.columns and "dt" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["dt"], errors="coerce")
        elif "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        else:
            frame["timestamp"] = pd.to_datetime(frame.index, errors="coerce")

        rename_map = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        frame = frame.rename(columns={src: dst for src, dst in rename_map.items() if src in frame.columns})
        for col in ("open", "high", "low", "close", "volume"):
            if col not in frame.columns:
                frame[col] = 0.0

        frame = frame.sort_values("timestamp", na_position="last").reset_index(drop=True)
        out: List[Dict[str, Any]] = []
        for idx, row in frame.iterrows():
            ts = pd.Timestamp(row.get("timestamp"))
            bar = {
                "idx": int(idx),
                "dt": ts.isoformat() if not pd.isna(ts) else "",
                "timestamp": int(ts.timestamp()) if not pd.isna(ts) else None,
                "o": self._float(row.get("open")),
                "h": self._float(row.get("high")),
                "l": self._float(row.get("low")),
                "c": self._float(row.get("close")),
                "v": self._float(row.get("volume")),
            }
            extra = {}
            for key in (
                "asset",
                "session",
                "regime_state_id",
                "structural_bias_6h",
                "flow_strength_1h",
                "confluence_score",
                "hazard_score",
                "zone_hi",
                "zone_lo",
                "fvg_mid_price",
            ):
                if key in row.index:
                    extra[key] = self._clean_value(row.get(key))
            if extra:
                bar["meta"] = extra
            out.append(bar)
        return out

    def _normalize_trades(self, trades: Any, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        frame = normalize_trade_frame(self._as_frame(trades))
        if frame.empty:
            return []

        time_index = [pd.Timestamp(bar["dt"]) for bar in bars if bar.get("dt")]
        out: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            entry_ts = pd.Timestamp(row.get("entry_ts")) if pd.notna(row.get("entry_ts")) else pd.NaT
            exit_ts = pd.Timestamp(row.get("exit_ts")) if pd.notna(row.get("exit_ts")) else pd.NaT
            rec = {
                "trade_id": str(row.get("trade_id") or ""),
                "asset": str(row.get("asset") or ""),
                "side": str(row.get("side") or "long"),
                "leg": str(row.get("leg") or "core"),
                "tier": str(row.get("tier") or "unranked"),
                "entry_dt": entry_ts.isoformat() if not pd.isna(entry_ts) else "",
                "exit_dt": exit_ts.isoformat() if not pd.isna(exit_ts) else "",
                "entry_idx": self._locate_bar(time_index, entry_ts),
                "exit_idx": self._locate_bar(time_index, exit_ts),
                "entry_price": self._float(row.get("entry_price")),
                "exit_price": self._float(row.get("exit_price")),
                "stop_price": self._float(row.get("stop_price")),
                "pnl": self._float(row.get("pnl")),
                "r": self._float(row.get("r")),
                "conf": self._float(row.get("conf")),
                "evr": self._float(row.get("evr")),
                "hazard_entry": self._float(row.get("hazard_entry")),
                "reason": str(row.get("reason") or ""),
                "session": str(row.get("session") or ""),
                "regime": str(row.get("regime") or ""),
            }
            out.append(rec)
        return out

    def _normalize_pnl(
        self,
        pnl: Any,
        trades: List[Dict[str, Any]],
        *,
        starting_equity: float,
    ) -> List[Dict[str, Any]]:
        frame = self._as_frame(pnl)
        if frame.empty and trades:
            trade_df = pd.DataFrame(trades)
            trade_df = trade_df.rename(columns={"entry_dt": "entry_ts"})
            trade_df["entry_ts"] = pd.to_datetime(trade_df["entry_ts"], errors="coerce")
            frame = build_equity_curve(trade_df, starting_equity=starting_equity)

        if not frame.empty:
            rename_map = {
                "dt": "timestamp",
                "time": "timestamp",
                "value": "equity",
                "pnl": "equity",
            }
            frame = frame.rename(columns={src: dst for src, dst in rename_map.items() if src in frame.columns})
            if "timestamp" in frame.columns:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            elif "entry_ts" in frame.columns:
                frame["timestamp"] = pd.to_datetime(frame["entry_ts"], errors="coerce")
            else:
                frame["timestamp"] = pd.NaT

            if "equity" not in frame.columns:
                first_numeric = next((c for c in frame.columns if c != "timestamp"), None)
                frame["equity"] = pd.to_numeric(frame[first_numeric], errors="coerce") if first_numeric else starting_equity
            frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce").ffill().fillna(starting_equity)
            if "drawdown" not in frame.columns:
                frame["drawdown"] = frame["equity"] - frame["equity"].cummax()

            frame = frame.sort_values("timestamp", na_position="last").reset_index(drop=True)
            return [
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat() if pd.notna(row["timestamp"]) else "",
                    "equity": self._float(row.get("equity")),
                    "drawdown": self._float(row.get("drawdown")),
                }
                for _, row in frame.iterrows()
            ]

        return [{"timestamp": "", "equity": float(starting_equity), "drawdown": 0.0}]

    def _normalize_reasoning(self, reasoning: Any, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        if reasoning is None:
            return {}
        if isinstance(reasoning, dict):
            out = {}
            for key, value in reasoning.items():
                ts = self._timestamp_key(key)
                out[ts] = self._clean_value(value)
            return out

        if isinstance(reasoning, Iterable) and not isinstance(reasoning, (str, bytes)):
            out = {}
            for row in reasoning:
                if not isinstance(row, dict):
                    continue
                ts = self._timestamp_key(
                    row.get("timestamp") or row.get("dt") or row.get("time") or row.get("entry_ts")
                )
                payload = {k: self._clean_value(v) for k, v in row.items() if k not in {"timestamp", "dt", "time"}}
                out[ts] = payload
            return out

        return {}

    @staticmethod
    def _as_frame(data: Any) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            if not data:
                return pd.DataFrame()
            if all(isinstance(v, list) for v in data.values()):
                try:
                    return pd.DataFrame(data)
                except Exception:
                    return pd.DataFrame([data])
            return pd.DataFrame([data])
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            return pd.DataFrame(list(data))
        return pd.DataFrame()

    @staticmethod
    def _locate_bar(time_index: List[pd.Timestamp], ts: pd.Timestamp) -> Optional[int]:
        if pd.isna(ts) or not time_index:
            return None
        idx = None
        for i, bar_ts in enumerate(time_index):
            if pd.isna(bar_ts):
                continue
            if bar_ts <= ts:
                idx = i
            else:
                break
        return idx

    @staticmethod
    def _timestamp_key(value: Any) -> str:
        try:
            ts = pd.Timestamp(value)
            return ts.isoformat() if not pd.isna(ts) else str(value)
        except Exception:
            return str(value)

    @staticmethod
    def _float(value: Any) -> float:
        try:
            val = float(value)
        except Exception:
            return 0.0
        if not math.isfinite(val):
            return 0.0
        return val

    def _clean_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._clean_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean_value(v) for v in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, str):
            return value
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:
            return float(value)
        except Exception:
            return str(value)
