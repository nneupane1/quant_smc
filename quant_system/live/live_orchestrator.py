"""
LiveOrchestrator
----------------
Real-time loop:
 - Streams 1m data (Kraken)
 - Aggregates to 15m bars
 - Runs model predictions
 - Confluence + EVR + Tiering
 - MPC risk + position sizing
 - Runner/core split + hazard trailing
"""

import uuid
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from quant_system.config.config_loader import ConfigLoader
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.tiering import TieringEngine
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.risk.mpc_risk import MPCRiskManager
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.live.kraken_live_client import KrakenLiveClient
from quant_system.live.live_executor import LiveExecutor, LivePosition
from quant_system.utils.logger import get_logger

LOG = get_logger("live_orchestrator")


class LiveOrchestrator:
    def __init__(self, config_loader: ConfigLoader, registry: ModelRegistry, dashboard_adapter=None):
        self.cfg_loader = config_loader
        self.cfg = config_loader.load()
        self.exec_cfg = self.cfg.get("execution", {})
        self.registry = registry
        self.dashboard = dashboard_adapter

        self.confluence = ConfluenceEngine(self.cfg)
        self.evr = EVRCalculator(self.cfg)
        self.gates = GateEvaluator(self.cfg)
        self.tiering = TieringEngine(self.cfg)
        self.hazard = HazardTrailingEngine(self.cfg)
        self.mpc = MPCRiskManager(self.cfg)
        self.position_sizer = PositionSizer(self.cfg)

        self.predictor = ModelPredictor(registry)
        self.models_loaded = False

        self.feed = KrakenLiveClient(self.cfg_loader)
        self.executor = LiveExecutor(config_loader)

        self.buffers = {"1m": []}
        self.bar_index = 0

        LOG.info("[LiveOrchestrator] Initialized.")

    # ------------------------------------------------------------------
    def load_models(self):
        # Predictor loads latest automatically via registry
        self.models_loaded = True

    # ------------------------------------------------------------------
    def run(self):
        LOG.info("[LiveOrchestrator] Live loop started.")
        if not self.models_loaded:
            self.load_models()

        for c1 in self.feed.run_stream():
            self.buffers["1m"].append(c1)
            if not self._ready_bar():
                continue

            bar = self._agg_15m()
            if bar is None:
                continue

            self.bar_index += 1
            asset = bar.get("asset", "BTCUSD")
            row = pd.Series(bar)

            # Hazard trailing on existing positions for this asset
            self._apply_hazard(asset, row)

            # Model inference + enrich
            enriched = self._inject_model_probs(row)

            # Confluence + EVR
            conf_result = self._compute_confluence(enriched)
            conf = conf_result.get("score", 0.0)
            conf_pass = conf_result.get("passed", False)
            enriched["confluence_score"] = conf

            side = enriched.get("side", "long")
            # TF gates (12h→6h→1h)
            gate = self.gates.evaluate(enriched, side)
            if not gate["passed"]:
                self._update_dashboard(enriched, conf, {"evr": None, "median_r": None})
                continue

            evr_pack = self.evr.compute_evr(enriched, side)
            hazard_score = enriched.get("hazard_score", enriched.get("hazard", 0.0))

            # Tiering
            tier = self.tiering.classify(
                row=enriched,
                confluence_pass=conf_pass,
                evr_result={"evr": evr_pack.get("evr"), "median_r": evr_pack.get("median_r")},
                hazard_score=hazard_score,
                bar_index=self.bar_index,
            )

            if tier.get("tier") == "skip" or not tier.get("execute", False):
                self._update_dashboard(enriched, conf, evr_pack)
                continue

            # MPC + sizing
            mpc_out = self.mpc.decide(
                equity=self.executor.equity,
                free_capital=self.executor.free_capital,
                locked_profit=self.executor.locked_profit,
                row=enriched,
            )

            stop_price = evr_pack.get("stop_price", enriched["close"])
            size_pack = self.position_sizer.size_position(
                row=enriched,
                equity=self.executor.equity - self.executor.locked_profit,
                side=side,
                stop_price=stop_price,
                risk_mode=mpc_out["risk_mode"],
                hedge_ratio=mpc_out["hedge_ratio"],
            )

            pos_usd = size_pack.get("value", 0.0)
            if pos_usd <= 0 or pos_usd > self.executor.free_capital:
                self._update_dashboard(enriched, conf, evr_pack)
                continue

            # Runner/core split
            split = self.exec_cfg.get("runner_split", {})
            core_frac = split.get("core_frac", 0.7)
            runner_frac = 1.0 - core_frac
            entry = enriched["close"]

            core_usd = pos_usd * core_frac
            if core_usd > 0:
                trade_id = f"{asset}-{uuid.uuid4().hex[:8]}-core"
                meta = {"leg": "core", "conf": conf, "tier": tier.get("tier"), "direction": side}
                self.executor.open_position(trade_id, asset, core_usd, entry, meta, stop_price=stop_price)

            if runner_frac > 0:
                runner_usd = pos_usd * runner_frac
                trade_id = f"{asset}-{uuid.uuid4().hex[:8]}-runner"
                meta = {"leg": "runner", "conf": conf, "tier": tier.get("tier"), "direction": side}
                self.executor.open_position(trade_id, asset, runner_usd, entry, meta, stop_price=stop_price)

            self._update_dashboard(enriched, conf, evr_pack)

    # ------------------------------------------------------------------
    def _apply_hazard(self, asset: str, row: pd.Series):
        to_close = []
        for tid, pos in list(self.executor.positions.items()):
            if pos.asset != asset:
                continue
            hazard_val = row.get("hazard_score", row.get("hazard", 0.0))
            trail = self.hazard.evaluate(
                row=row,
                hazard=hazard_val,
                side=pos.side,
                bars_in_trade=pos.bars_in_trade,
                current_stop=pos.stop_price or row["close"],
                config=self.cfg,
                leg=pos.meta.get("leg", "core"),
            )
            pos.bars_in_trade += 1
            if trail["action"] == "exit":
                self.executor.exit_position(tid, price=trail["new_stop"])
                to_close.append(tid)
            elif trail["action"] in ("partial", "tighten"):
                pos.stop_price = trail["new_stop"]

        for tid in to_close:
            self.executor.positions.pop(tid, None)

    # ------------------------------------------------------------------
    def _compute_confluence(self, row: pd.Series) -> Dict[str, Any]:
        out = self.confluence.evaluate(row)
        score = out.get("confluence_score", out.get("score", 0.0))
        allow = out.get("passed", out.get("allow", False))
        return {"score": score, "passed": allow}

    def _ready_bar(self) -> bool:
        if len(self.buffers["1m"]) < 15:
            return False
        last_ts = self.buffers["1m"][-1]["timestamp"]
        return last_ts % 900 == 0

    def _agg_15m(self) -> Optional[Dict[str, Any]]:
        rows = self.buffers["1m"][-15:]
        if len(rows) < 15:
            return None
        o = rows[0]["open"]
        h = max(r["high"] for r in rows)
        l = min(r["low"] for r in rows)
        c = rows[-1]["close"]
        v = sum(r["volume"] for r in rows)
        return {
            "timestamp": rows[-1]["timestamp"],
            "dt": datetime.utcfromtimestamp(rows[-1]["timestamp"]),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "asset": rows[-1].get("asset", "BTCUSD"),
        }

    def _inject_model_probs(self, row: pd.Series) -> pd.Series:
        enriched = row.copy()
        try:
            feats = [v for _, v in row.items() if isinstance(v, (int, float))]
            specialist_list = ["liq_flow", "bos_cont", "momo", "eop", "edp"]
            preds = self.predictor.predict_single(feats, specialist_list)
            for k, v in preds.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        enriched[f"{k}_{kk}"] = vv
                else:
                    if k.startswith("prob_"):
                        enriched[f"p_{k.replace('prob_', '')}"] = v
                    else:
                        enriched[k] = v
            if "hazard_score" in preds:
                enriched["hazard"] = preds["hazard_score"]
        except Exception as e:
            LOG.warning(f"[LiveOrchestrator] predictor failed: {e}")
        return enriched

    def _update_dashboard(self, row: pd.Series, conf: float, evr: Dict[str, Any]):
        if not self.dashboard:
            return
        self.dashboard.update_state({
            "timestamp": row.get("dt"),
            "equity": self.executor.equity,
            "free_capital": self.executor.free_capital,
            "locked_profit": getattr(self.executor, "locked_profit", 0),
            "confluence": conf,
            "evr": evr.get("evr"),
            "positions": len(self.executor.positions),
        })
