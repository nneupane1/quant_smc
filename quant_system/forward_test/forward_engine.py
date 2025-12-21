"""
forward_engine.py — full multi-asset forward engine.

Runs as a real-time machine:
 - listens to 1m price feed
 - aggregates to 15m / 1h / 6h / 12h
 - evaluates SMC + ML + tiering
 - supports multi-asset switching on-the-fly
 - executes real orders via LiveExecutor
 - hazard trailing + MPC risk engine
 - compounding logic + vault lock
 - adaptive cooling + moonshot override
 - dashboard streaming (equity, trades, reasoning)

This engine is the live mirror of the backtester.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.config.config_loader import ConfigLoader

from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.tiering import TieringEngine
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.risk.mpc_risk import MPCRiskManager as MPCRiskEngine
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.execution.risk.exposure_tracker import ExposureTracker

from quant_system.forward_test.forward_executor import ForwardExecutor
from quant_system.forward_test.forward_reasoning_attach import ReasoningAttach
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.utils.logger import get_logger


LOG = get_logger("forward_engine")


class ForwardEngine:
    """
    Live/paper multi-asset engine with full trading logic.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        model_registry: ModelRegistry,
        dashboard_adapter=None,
        execution_bridge=None
    ):
        self.cfg = config_loader
        self.cfg_loader = config_loader
        self.registry = model_registry
        self.assets_cfg = self.cfg.load_yaml("assets.yaml")
        self.exec_cfg = self.cfg.load_yaml("execution.yaml")

        self.models = None
        self.dashboard = dashboard_adapter

        self.confluence = ConfluenceEngine(config_loader)
        self.evr = EVRCalculator(config_loader)
        self.gates = GateEvaluator(config_loader)
        self.tiering = TieringEngine(config_loader)
        self.hazard = HazardTrailingEngine(config_loader)
        self.mpc = MPCRiskEngine(config_loader)
        self.position_sizer = PositionSizer()
        self.exposure = ExposureTracker()
        self.predictor = ModelPredictor(model_registry)

        self.executor = ForwardExecutor(config_loader)

        self.reason = ReasoningAttach()

        # Live state
        self.equity = self.exec_cfg["starting_equity"]
        self.free_capital = self.equity
        self.locked_profit = 0
        self.max_equity = self.equity
        self.cooling_until = None

        self.open_positions = {}
        self.trade_log = []

        LOG.info("[ForwardEngine] Initialized with multi-asset support")

    # ----------------------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------------------
    def load_models(self, version: str):
        """
        Load latest generic model artifacts for forward execution.
        """
        names = [
            "liq_flow",
            "bos_cont",
            "momo",
            "eop",
            "edp",
            "meta_model",
            "confluence_model",
            "hazard",
            "quantile",
        ]
        self.models = {}
        for name in names:
            try:
                self.models[name], _ = self.registry.load_latest(name)
                LOG.info(f"[ForwardEngine] Loaded model {name}")
            except Exception as e:
                LOG.warning(f"[ForwardEngine] Missing model {name}: {e}")
        self.predictor = ModelPredictor(self.registry)
        if self.dashboard:
            self.dashboard.log_event("models_loaded", None, {"version": version})

    # ----------------------------------------------------------------------
    # MAIN ENTRYPOINT — receives aggregated TF rows (15m close)
    # ----------------------------------------------------------------------
    def on_bar(self, asset: str, row: Dict[str, Any]):
        """
        Called at every 15m bar close per asset.
        row is a dict-like structure with SMC + features + timestamp.
        """

        if self.models is None:
            LOG.warning("[ForwardEngine] Ignoring bar: models not loaded")
            return

        dt = row["dt"]

        # Dashboard candle push
        if self.dashboard:
            self.dashboard.update_candles({asset: row})

        # Cooling logic
        if self.cooling_until and dt < self.cooling_until:
            self._update_equity(dt, row)
            return

        # Hazard trailing first
        # Hazard trailing on existing positions before new entry decisions
        self._apply_hazard(asset, row, dt)

        # Confluence + EVR
        enriched_row = self._inject_model_probs(row)
        side = enriched_row.get("side", "long")

        gate = self.gates.evaluate(enriched_row, side)
        if not gate["passed"]:
            self._update_equity(dt, row)
            return

        cdata = self.confluence.compute(self.models, enriched_row)
        conf = cdata["conf_score"]

        evr = self.evr.compute(self.models, enriched_row)

        # Tiering
        tier = self.tiering.decide(conf, evr)

        # Moonshot override
        if self._moonshot_override(row, conf, evr):
            tier = "A+"

        if tier == "skip":
            self._update_equity(dt, row)
            return

        # MPC risk logic
        mpc_out = self.mpc.compute(self.equity, self.free_capital, self.locked_profit, row)
        risk = mpc_out["risk_mode"]
        hedge_ratio = mpc_out["hedge_ratio"]
        lock_fraction = mpc_out["lock_fraction"]

        # Update locked profit
        if lock_fraction > 0:
            self._lock_profit(lock_fraction)

        # Determine position size
        pos_usd = self.position_sizer.size(self.equity - self.locked_profit, risk)
        if pos_usd <= 0 or pos_usd > self.free_capital:
            self._update_equity(dt, row)
            return

        # Execute order
        entry_price = row["close"]
        # Split into core and runner legs
        core_frac = self.exec_cfg.get("runner_split", {}).get("core_frac", 0.7)
        runner_frac = 1.0 - core_frac

        core_notional = pos_usd * core_frac
        runner_notional = pos_usd * runner_frac

        pos_core = self.executor.open_position(asset, entry_price, core_notional, row, leg="core")
        self.free_capital -= core_notional
        self.open_positions[pos_core.trade_id] = pos_core

        if runner_notional > 0:
            pos_runner = self.executor.open_position(asset, entry_price, runner_notional, row, leg="runner")
            self.free_capital -= runner_notional
            self.open_positions[pos_runner.trade_id] = pos_runner

        # Reasoning logging
        reason = self.reason.build(asset, row, conf, evr, tier, risk, hedge_ratio)
        self.trade_log.append({"open": pos_core, "reason": reason})

        if self.dashboard:
            self.dashboard.log_event("entry", pos_core.trade_id, reason)
            if runner_notional > 0:
                self.dashboard.log_event("entry", pos_runner.trade_id, {**reason, "leg": "runner"})

        self._update_equity(dt, row)

    # ----------------------------------------------------------------------
    def _inject_model_probs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs specialist/meta/hazard inference on the current feature row and attaches probabilities.
        """
        enriched = dict(row)
        try:
            feats = [v for k, v in row.items() if isinstance(v, (int, float))]
            specialist_list = ["liq_flow", "bos_cont", "momo", "eop", "edp"]
            preds = self.predictor.predict_single(feats, specialist_list)
            # flatten preds into row
            for k, v in preds.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        enriched[f"{k}_{kk}"] = vv
                else:
                    enriched[k] = v
        except Exception as e:
            LOG.warning(f"[ForwardEngine] predictor failed: {e}")
        return enriched

    # ----------------------------------------------------------------------
    # HAZARD TRAILING FOR OPEN POSITIONS
    # ----------------------------------------------------------------------
    def _apply_hazard(self, asset, row, dt):
        if "hazard" not in self.models:
            return

        to_close = []
        price = row["close"]
        hazard_score = row.get("hazard_score")

        # If hazard_score not in row, compute via predictor using numeric features
        if hazard_score is None:
            try:
                feats = [v for k, v in row.items() if isinstance(v, (int, float))]
                preds = self.predictor.predict_single(feats, [])
                hazard_score = preds.get("hazard_score", 0.0)
            except Exception:
                hazard_score = 0.0

        for tid, pos in list(self.open_positions.items()):
            if pos.asset != asset:
                continue

            # bars in trade approx by elapsed time / 15m
            bars_in_trade = int(max(1, (dt - pos.opened_at).total_seconds() // (15 * 60)))

            decision = self.hazard.evaluate(
                row=pd.Series(row),
                hazard=float(hazard_score or 0.0),
                side=getattr(pos, "side", "long"),
                bars_in_trade=bars_in_trade,
                current_stop=getattr(pos, "entry_price", price),
                config=self.cfg_loader.load(),
                leg=getattr(pos, "leg", "core")
            )

            action = decision["action"]
            new_stop = decision["new_stop"]

            if action in ("tighten", "partial"):
                pos.entry_price = new_stop  # proxy: treat as tightened stop
            if action == "exit":
                res = self.executor.exit_position(pos, price)
                self.free_capital += res["value"]
                to_close.append((tid, res))

        for tid, res in to_close:
            del self.open_positions[tid]
            if self.dashboard:
                self.dashboard.log_event("exit", tid, {"pnl": res["pnl"], "value": res["value"]})

    # ----------------------------------------------------------------------
    # EQUITY UPDATE + COOLING CHECK
    # ----------------------------------------------------------------------
    def _update_equity(self, dt, row=None):
        price = row["close"] if row else None
        mtm = sum(self.executor.mark_to_market(p, price or p.entry_price) for p in self.open_positions.values())
        self.equity = self.locked_profit + self.free_capital + mtm

        self.max_equity = max(self.max_equity, self.equity)

        dd = (self.max_equity - self.equity) / max(self.max_equity, 1e-9)
        if dd >= abs(self.exec_cfg.get("cooling_dd_trigger", 0.0)):
            self.cooling_until = dt + timedelta(minutes=self.exec_cfg.get("cooling_minutes", 0))

        if self.dashboard:
            self.dashboard.update_state({
                "timestamp": dt,
                "equity": self.equity,
                "free_capital": self.free_capital,
                "locked_profit": self.locked_profit,
                "max_drawdown": dd,
                "open_positions": len(self.open_positions)
            })

    # ----------------------------------------------------------------------
    # LOCK PROFIT INTO VAULT
    # ----------------------------------------------------------------------
    def _lock_profit(self, frac):
        delta = (self.equity - self.locked_profit) * frac
        if delta > 0:
            self.locked_profit += delta
            self.free_capital -= delta

    # ----------------------------------------------------------------------
    # MOONSHOT OVERRIDE LOGIC
    # ----------------------------------------------------------------------
    def _moonshot_override(self, row, conf, evr):
        """
        If the setup is extremely strong, do not miss it.
        """
        if conf >= 0.92 and evr["median_R"] >= 6.0:
            return True

        if evr["expected_R"] >= 3.0 and evr["p_big_run"] >= 0.40:
            return True

        return False
