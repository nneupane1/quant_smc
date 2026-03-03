"""
Replay Controller
Builds candle-by-candle payloads for backtest replay from the current backtest
engine outputs and current model/gating stack.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from quant_system.backtest.replay.replay_timeline import build_timeline
from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.risk.mpc_risk import MPCRiskManager
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.utils.logger import get_logger

LOG = get_logger("replay_controller")


@dataclass
class ReplayState:
    ptr: int = 0
    playing: bool = False
    last_payload: Optional[Dict[str, Any]] = field(default_factory=dict)


class ReplayController:
    def __init__(
        self,
        candles_15m: pd.DataFrame,
        smc_features: Optional[pd.DataFrame],
        execution_log: Optional[pd.DataFrame],
        model_bundle: Optional[Any],
        config: Dict[str, Any],
    ):
        self.df = candles_15m.reset_index(drop=True).copy()
        if "timestamp" not in self.df.columns and "dt" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["dt"], errors="coerce")
        elif "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

        self.smc = (smc_features.reset_index(drop=True).copy() if smc_features is not None else self.df.copy())
        if "timestamp" not in self.smc.columns and "dt" in self.smc.columns:
            self.smc["timestamp"] = pd.to_datetime(self.smc["dt"], errors="coerce")

        self.exec = build_timeline(execution_log, candles=self.df)
        self.state = ReplayState()
        self.config = config

        self.confluence = ConfluenceEngine(config)
        self.gates = GateEvaluator(config)
        self.hazard_engine = HazardTrailingEngine(config)
        self.mpc_risk = MPCRiskManager(config)
        self.evr_calc = EVRCalculator(config)

        self.models = model_bundle
        self.predictor = self._coerce_predictor(model_bundle)
        self.n = len(self.df)

    @staticmethod
    def _coerce_predictor(model_bundle: Any) -> Optional[ModelPredictor]:
        if isinstance(model_bundle, ModelPredictor):
            return model_bundle
        if isinstance(model_bundle, ModelRegistry):
            return ModelPredictor(model_bundle)
        return None

    @staticmethod
    def _session_name(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "other"
        hour = ts.hour
        if 7 <= hour < 13:
            return "london"
        if 13 <= hour < 16:
            return "overlap"
        if 16 <= hour < 21:
            return "ny"
        return "other"

    @staticmethod
    def _parse_jsonish(value):
        if isinstance(value, (dict, list)):
            return value
        if value is None:
            return []
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def jump_to(self, idx: int):
        idx = max(0, min(idx, self.n - 1))
        self.state.ptr = idx
        self.state.last_payload = self.render_payload()
        LOG.info("Replay jump to idx=%s", idx)

    def jump_to_timestamp(self, ts: pd.Timestamp):
        if self.df.empty:
            return
        ts = pd.Timestamp(ts)
        mask = self.df["timestamp"] >= ts
        idx = int(mask.idxmax()) if mask.any() else self.n - 1
        self.jump_to(idx)

    def step_forward(self):
        if self.state.ptr < self.n - 1:
            self.state.ptr += 1
        return self.render_payload()

    def step_backward(self):
        if self.state.ptr > 0:
            self.state.ptr -= 1
        return self.render_payload()

    def render_payload(self) -> Dict[str, Any]:
        if self.df.empty:
            payload = {"candle": {}, "smc": {}, "trade": {}, "meta": {}}
            self.state.last_payload = payload
            return payload

        i = self.state.ptr
        row = self.df.iloc[i].copy()
        smc_row = self.smc.iloc[i].copy() if i < len(self.smc) else pd.Series(dtype=object)
        row_dict = row.to_dict()
        if not row_dict.get("session"):
            row_dict["session"] = self._session_name(pd.Timestamp(row_dict.get("timestamp")))

        prob_dict = self._model_predict(row_dict)
        merged_row = {**row_dict, **prob_dict}
        evr_out = self._safe_evr(merged_row)
        conf_out = self.confluence.evaluate(pd.Series(merged_row))
        hazard = float(prob_dict.get("hazard_score", row_dict.get("hazard", 0.0)) or 0.0)
        side = str(row_dict.get("side", "long") or "long")
        gate = self.gates.evaluate(pd.Series(merged_row), side)
        risk_state = self.mpc_risk.decide(
            equity=float(row_dict.get("equity", 0.0) or 0.0),
            free_capital=float(row_dict.get("free_capital", row_dict.get("equity", 0.0)) or 0.0),
            locked_profit=float(row_dict.get("locked_profit", 0.0) or 0.0),
            row=merged_row,
        )

        payload = {
            "candle": {
                "time": int(pd.Timestamp(row["timestamp"]).timestamp()) if pd.notna(row["timestamp"]) else None,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            },
            "smc": self._build_smc_payload(smc_row),
            "trade": self._extract_trade_markers(i),
            "meta": {
                "idx": int(i),
                "timestamp": str(row["timestamp"]),
                "conf": float(conf_out.get("confluence_score", conf_out.get("score", 0.0)) or 0.0),
                "evr": float(evr_out.get("evr", 0.0) or 0.0),
                "median_r": float(evr_out.get("median_r", 0.0) or 0.0),
                "hazard": hazard,
                "risk": risk_state,
                "gates": gate,
                "probabilities": prob_dict,
            },
        }
        self.state.last_payload = payload
        return payload

    def _build_smc_payload(self, smc_row: pd.Series) -> Dict[str, Any]:
        payload = {
            "swings": self._parse_jsonish(smc_row.get("swings_json", [])),
            "orderblocks": self._parse_jsonish(smc_row.get("ob_json", [])),
            "fvg": self._parse_jsonish(smc_row.get("fvg_json", [])),
            "sweeps": self._parse_jsonish(smc_row.get("sweeps_json", [])),
            "bos_choch": self._parse_jsonish(smc_row.get("bos_choch_json", [])),
        }
        if not any(payload.values()):
            payload = {
                "swings": {
                    "swing_high": smc_row.get("swing_high"),
                    "swing_low": smc_row.get("swing_low"),
                },
                "orderblocks": {
                    "demand_zone": smc_row.get("demand_zone"),
                    "supply_zone": smc_row.get("supply_zone"),
                },
                "fvg": {
                    "fvg_mid": smc_row.get("fvg_mid"),
                    "fvg_open": smc_row.get("fvg_open"),
                },
                "sweeps": {
                    "sweep_high": smc_row.get("sweep_high"),
                    "sweep_low": smc_row.get("sweep_low"),
                },
                "bos_choch": {
                    "bos_up": smc_row.get("bos_up", smc_row.get("bos_flag")),
                    "bos_down": smc_row.get("bos_down"),
                    "choch": smc_row.get("choch_flag"),
                },
            }
        return payload

    def _extract_trade_markers(self, idx: int) -> Dict[str, Any]:
        if self.exec.empty:
            return {"entries": [], "exits": [], "stops": [], "hedge": []}

        subset = self.exec[self.exec.get("candle_idx").eq(idx)] if "candle_idx" in self.exec.columns else pd.DataFrame()
        if subset.empty and "timestamp" in self.exec.columns:
            ts = self.df.iloc[idx]["timestamp"]
            subset = self.exec[self.exec["timestamp"] == ts]

        entries = []
        exits = []
        stops = []
        hedge = []
        for _, row in subset.iterrows():
            rec = {
                "trade_id": row.get("trade_id"),
                "price": float(row.get("price", 0.0) or 0.0),
                "side": row.get("side"),
                "reason": row.get("reason"),
                "leg": row.get("leg"),
            }
            event_type = row.get("type")
            if event_type == "entry":
                entries.append(rec)
            elif event_type == "stop":
                stops.append(rec)
            elif event_type == "hedge":
                hedge.append(rec)
            else:
                exits.append(rec)
        return {"entries": entries, "exits": exits, "stops": stops, "hedge": hedge}

    def _model_predict(self, row_like: Dict[str, Any]) -> Dict[str, Any]:
        specialist_list = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
        if self.predictor is not None:
            try:
                return self.predictor.predict_single(row_like, specialist_list)
            except Exception as exc:
                LOG.warning("Replay predictor failed: %s", exc)

        out: Dict[str, Any] = {}
        if isinstance(self.models, dict):
            for name in specialist_list:
                model = self.models.get(name)
                if model is None:
                    continue
                try:
                    p = model.predict_proba(pd.DataFrame([row_like]))[0][1]
                except Exception:
                    p = 0.0
                out[f"prob_{name}"] = float(p)
        out.setdefault("hazard_score", float(row_like.get("hazard", 0.0) or 0.0))
        return out

    def _safe_evr(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.evr_calc.compute_evr(pd.Series(row), row.get("side", "long"))
        except Exception:
            return {"evr": 0.0, "median_r": 0.0, "stop_price": row.get("close")}

    def export_last_json(self):
        return json.dumps(self.state.last_payload or {})
