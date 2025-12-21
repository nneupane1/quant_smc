"""
Replay Controller
Feeds candles + overlays + execution info into the TradingView replay widget.

This is used by:
 - Streamlit replay_mode.py (UI)
 - Backtest visualization extension
 - Internal flight-recorder debugging

Design:
 - Maintain replay state: pointer index, dataframe slices
 - Serve next / prev candle updates
 - Produce JSON payload for JS widget: candles, overlays, trade markers
 - Includes evolving Confluence, Hazard, EVR, SMC, MPC risk state.

Replay format sent to JS (one update per step):
{
    "candle": {...},
    "smc": {...},
    "trade": {...},
    "risk": {...},
    "meta": {...}
}
"""

import pandas as pd
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from quant_system.utils.logger import get_logger
from quant_system.execution.confluence import ConfluenceEngine
from quant_system.execution.hazard_trailing import HazardTrailingEngine
from quant_system.execution.mpc_risk import MPCRiskManager
from quant_system.execution.evr import EVRCalculator
try:
    from quant_system.execution.sessions import SessionClassifier
except Exception:
    class SessionClassifier:  # fallback placeholder
        def __init__(self, *_args, **_kwargs):
            pass

        def classify(self, ts):
            return "other"

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
        smc_features: pd.DataFrame,
        execution_log: pd.DataFrame,
        model_bundle: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """
        candles_15m: OHLCV + indicators + timestamps (sorted ascending)
        smc_features: SMC structures for each candle index
        execution_log: trade markers (entry, exit, stop, etc.)
        model_bundle: dict of loaded models (liq_flow, bos_cont, momo, meta, hazard, quantile)
        config: config dict loaded from YAML
        """
        self.df = candles_15m.reset_index(drop=True)
        self.smc = smc_features.reset_index(drop=True)
        self.exec = execution_log.reset_index(drop=True)
        self.state = ReplayState()
        self.config = config

        # Engines
        self.confluence = ConfluenceEngine(config)
        self.hazard_engine = HazardTrailingEngine(config)
        self.mpc_risk = MPCRiskManager(config)
        self.evr_calc = EVRCalculator(config)
        self.session = SessionClassifier(config)

        self.models = model_bundle
        self.n = len(self.df)

    # ---------------------------------------------------------
    # Pointer Controls
    # ---------------------------------------------------------

    def jump_to(self, idx: int):
        """Jump replay pointer to a given candle index."""
        idx = max(0, min(idx, self.n - 1))
        self.state.ptr = idx
        LOG.info(f"Replay jump to idx={idx}")

    def jump_to_timestamp(self, ts: pd.Timestamp):
        """Jump replay pointer to the candle nearest timestamp."""
        idx = int(self.df.index[self.df['timestamp'] >= ts][0]) if (self.df['timestamp'] >= ts).any() else 0
        self.jump_to(idx)

    def step_forward(self):
        """Move forward by 1 candle."""
        if self.state.ptr < self.n - 1:
            self.state.ptr += 1
        LOG.info(f"Replay step → idx={self.state.ptr}")
        return self.render_payload()

    def step_backward(self):
        """Move backward by 1 candle."""
        if self.state.ptr > 0:
            self.state.ptr -= 1
        LOG.info(f"Replay step back → idx={self.state.ptr}")
        return self.render_payload()

    # ---------------------------------------------------------
    # Build Replay Payload
    # ---------------------------------------------------------

    def render_payload(self) -> Dict[str, Any]:
        """Builds the full replay JSON payload for the current index."""
        i = self.state.ptr
        row = self.df.iloc[i]
        smc_row = self.smc.iloc[i]

        # -----------------------------------------------------
        # 1) Candle Payload
        # -----------------------------------------------------
        candle = {
            "time": int(pd.Timestamp(row["timestamp"]).timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }

        # -----------------------------------------------------
        # 2) SMC Overlays
        # -----------------------------------------------------
        smc_payload = {
            "swings": json.loads(smc_row.get("swings_json", "[]")),
            "orderblocks": json.loads(smc_row.get("ob_json", "[]")),
            "fvg": json.loads(smc_row.get("fvg_json", "[]")),
            "sweeps": json.loads(smc_row.get("sweeps_json", "[]")),
            "bos_choch": json.loads(smc_row.get("bos_choch_json", "[]")),
        }

        # -----------------------------------------------------
        # 3) Trade Markers
        # -----------------------------------------------------
        trade_payload = self._extract_trade_markers(i)

        # -----------------------------------------------------
        # 4) Confluence / EVR / Hazard Models
        # -----------------------------------------------------
        features = self._extract_feature_vector(i)
        prob_dict = self._model_predict(features)

        conf = self._safe_confluence(row, smc_row, prob_dict)
        evr = self._safe_evr(prob_dict, row)
        hazard = float(row.get("hazard", prob_dict.get("hazard", 0.0)))

        # -----------------------------------------------------
        # 5) MPC Risk State
        # -----------------------------------------------------
        try:
            risk_state = self.mpc_risk.compute_from_snapshot(prob_dict, row)  # type: ignore[attr-defined]
        except Exception:
            risk_state = {}

        # -----------------------------------------------------
        # 6) Meta Info
        # -----------------------------------------------------
        meta = {
            "idx": int(i),
            "timestamp": str(row["timestamp"]),
            "conf": float(conf),
            "evr": float(evr),
            "hazard": float(hazard),
            "risk": risk_state,
        }

        payload = {
            "candle": candle,
            "smc": smc_payload,
            "trade": trade_payload,
            "meta": meta,
        }

        self.state.last_payload = payload
        return payload

    # ---------------------------------------------------------
    # Trade Markers
    # ---------------------------------------------------------

    def _extract_trade_markers(self, idx: int) -> Dict[str, Any]:
        """Return trades that occur at this candle index."""
        subset = self.exec[self.exec["candle_idx"] == idx]
        entries = []
        exits = []
        stops = []
        hedge = []

        for _, row in subset.iterrows():
            if row["type"] == "entry":
                entries.append({"price": float(row["price"]), "side": row["side"]})
            elif row["type"] == "exit":
                exits.append({"price": float(row["price"]), "reason": row["reason"]})
            elif row["type"] == "stop":
                stops.append({"price": float(row["price"])})
            elif row["type"] == "hedge":
                hedge.append({"ratio": float(row["ratio"]), "price": float(row["price"])})

        return {"entries": entries, "exits": exits, "stops": stops, "hedge": hedge}

    # ---------------------------------------------------------
    # Feature Vector for ML Models
    # ---------------------------------------------------------

    def _extract_feature_vector(self, idx: int):
        """Extract model-ready input vector from row."""
        row = self.df.iloc[idx]
        feat_cols = [c for c in self.df.columns if c.startswith("feat_")]
        return row[feat_cols].values.reshape(1, -1)

    def _model_predict(self, features) -> Dict[str, float]:
        """Run all specialist models."""
        out = {}

        for name, model in self.models.items():
            try:
                p = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else float(model.predict(features))
            except Exception:
                p = 0.0
            out[name] = float(p)

        return out

    def _safe_confluence(self, row, smc_row, prob_dict) -> float:
        try:
            if hasattr(self.confluence, "evaluate"):
                res = self.confluence.evaluate(row)
                return res.get("confluence_score") or res.get("score") or 0.0
            if hasattr(self.confluence, "compute_confluence"):
                res = self.confluence.compute_confluence(smc_row, prob_dict, row, self.config)  # type: ignore[attr-defined]
                return res.get("score", 0.0)
        except Exception:
            return 0.0
        return 0.0

    def _safe_evr(self, prob_dict, row):
        try:
            if hasattr(self.evr_calc, "compute"):
                return self.evr_calc.compute(prob_dict, row)  # type: ignore[attr-defined]
            if hasattr(self.evr_calc, "compute_evr"):
                return self.evr_calc.compute_evr(row, row.get("side", "long"))
        except Exception:
            return 0.0
        return 0.0

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    def export_last_json(self):
        """Return last replay payload as JSON string for JS widget."""
        return json.dumps(self.state.last_payload or {})
