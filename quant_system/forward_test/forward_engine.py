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
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.config.config_loader import ConfigLoader

from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.tiering import TieringEngine
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.gating.profit_ladder import ProfitLadderManager
from quant_system.execution.risk.mpc_risk import MPCRiskManager as MPCRiskEngine
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.execution.risk.exposure_tracker import ExposureTracker
from quant_system.execution.risk.capital_allocator import CapitalAllocator
from quant_system.execution.risk.compound_cooling import CompoundCoolingPolicy

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
        merged_cfg = self.cfg.load()
        self.assets_cfg = self.cfg.load_yaml("assets.yaml")
        self.exec_cfg = merged_cfg.get("execution", {})
        self.manual_alert_only = bool(self.exec_cfg.get("manual_alert_only", False))

        self.models = None
        self.dashboard = dashboard_adapter

        self.confluence = ConfluenceEngine(config_loader)
        self.evr = EVRCalculator(config_loader)
        self.gates = GateEvaluator(config_loader)
        self.tiering = TieringEngine(config_loader)
        self.hazard = HazardTrailingEngine(config_loader)
        self.profit_ladder = ProfitLadderManager(config_loader)
        self.mpc = MPCRiskEngine(config_loader)
        self.capital = CapitalAllocator(config_loader)
        self.compound_cooling = CompoundCoolingPolicy(config_loader)
        self.position_sizer = PositionSizer()
        self.exposure = ExposureTracker(merged_cfg)
        self.predictor = ModelPredictor(model_registry)

        self.executor = ForwardExecutor(config_loader)

        self.reason = ReasoningAttach()

        # Live state
        self.equity = float(self.exec_cfg.get("starting_equity", 0.0))
        self.free_capital = self.equity
        self.locked_profit = 0
        self.max_equity = self.equity
        self.current_drawdown = 0.0
        self.cooling_until = None
        self.current_risk_mode = None
        self.current_hedge_ratio = 0.0
        self.bar_index = 0
        self.last_timestamp = None
        self.last_prices: Dict[str, float] = {}
        self.max_drawdown_seen = 0.0

        self.open_positions = {}
        self.closed_trades: Dict[str, Dict[str, Any]] = {}
        self.trade_log = []
        self.entry_reason_by_trade: Dict[str, Dict[str, Any]] = {}

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
            "flow_1h",
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
        self.last_timestamp = dt
        self.last_prices[asset] = float(row["close"])
        self.bar_index += 1

        # Dashboard candle push
        if self.dashboard:
            self.dashboard.update_candles({asset: row})

        enriched_row = self._inject_model_probs(row)
        had_ladder_exit = self._apply_profit_ladder(asset, enriched_row, dt)
        had_hazard_exit = self._apply_hazard(asset, enriched_row, dt)
        self._update_equity(dt, enriched_row)
        if had_ladder_exit or had_hazard_exit:
            return
        danger = self.compound_cooling.evaluate_danger(
            dt=dt,
            equity=self.equity,
            free_capital=self.free_capital,
            locked_profit=self.locked_profit,
            drawdown=self.current_drawdown,
            row=enriched_row,
            cooling_until=self.cooling_until,
        )
        if danger["lock_amount"] > 0:
            self._lock_profit_amount(danger["lock_amount"])
        if danger["cooling_until"] is not None:
            self.cooling_until = danger["cooling_until"]
        side = enriched_row.get("side", "long")

        gate = self.gates.evaluate(enriched_row, side)
        if not gate["passed"]:
            self._update_equity(dt, enriched_row)
            return

        cdata = self.confluence.compute(self.models, enriched_row)
        conf = cdata["conf_score"]
        enriched_row["confluence_score"] = conf
        enriched_row["conf_score"] = conf

        evr = self.evr.compute(self.models, enriched_row)

        # Tiering
        tier_result = self.tiering.classify(
            row=pd.Series(enriched_row),
            confluence_pass=bool(cdata.get("passed", False)),
            evr_result=evr,
            hazard_score=float(danger["metrics"]["hazard"] or 0.0),
            bar_index=self.bar_index,
        )

        # Moonshot override
        if self._moonshot_override(row, conf, evr):
            tier_result = {"tier": "A+", "execute": True, "reason": "moonshot_override"}

        cooling_gate = self.compound_cooling.allow_entry(
            now=dt,
            cooling_until=self.cooling_until,
            conf=conf,
            evr=evr,
            hazard=danger["metrics"]["hazard"],
        )
        if not cooling_gate["allow"]:
            self._update_equity(dt, enriched_row)
            return

        if tier_result.get("tier") == "skip" or not tier_result.get("execute", False):
            self._update_equity(dt, enriched_row)
            return

        if self.manual_alert_only:
            self._emit_manual_alert(
                asset=asset,
                dt=dt,
                row=enriched_row,
                conf=float(conf),
                evr=evr,
                tier_result=tier_result,
                gate=gate,
                danger=danger,
                cooling_gate=cooling_gate,
            )
            self._update_equity(dt, enriched_row)
            return

        capital_out = self.capital.allocate(
            equity=self.equity,
            free_capital=self.free_capital,
            locked_profit=self.locked_profit,
            row=enriched_row,
            mpc_manager=self.mpc,
        )
        risk = capital_out["risk_mode"]
        hedge_ratio = capital_out["hedge_ratio"]
        lock_fraction = capital_out["lock_fraction"]
        base_ticket = float(capital_out.get("ticket_usd", 0.0))
        if risk is None:
            risk = float(np.clip(base_ticket / max(self.equity, 1e-9), 0.0, 1.0))
        self.current_risk_mode = risk
        self.current_hedge_ratio = hedge_ratio

        # Update locked profit
        if lock_fraction > 0:
            self._lock_profit(lock_fraction)

        # Determine position size (allocator ticket capped by dynamic position-sizer value).
        sizing = self.position_sizer.size_position(
            row=pd.Series(enriched_row),
            equity=max(self.equity, 0.0),
            side=side,
            stop_price=float(enriched_row.get("stop_price", row["close"])),
            risk_mode=float(risk),
            hedge_ratio=float(hedge_ratio),
        )
        pos_usd = min(float(capital_out["ticket_usd"]), float(sizing.get("value", 0.0)), float(self.free_capital))
        enriched_row["sizer_value_usd"] = float(sizing.get("value", 0.0))
        enriched_row["sizer_qty"] = float(sizing.get("qty", 0.0))
        enriched_row["sizer_risk_dollars"] = float(sizing.get("risk_dollars", 0.0))
        enriched_row["position_notional_usd"] = float(pos_usd)
        if pos_usd <= 0 or pos_usd > self.free_capital:
            self._update_equity(dt, enriched_row)
            return

        # Execute order
        enriched_row["evr"] = evr.get("evr")
        enriched_row["median_r"] = evr.get("median_r")
        enriched_row["stop_price"] = evr.get("stop_price", row["close"])
        entry_price = row["close"]
        # Split into core and runner legs
        core_frac = self.exec_cfg.get("runner_split", {}).get("core_frac", 0.7)
        runner_frac = 1.0 - core_frac

        core_notional = pos_usd * core_frac
        runner_notional = pos_usd * runner_frac

        pos_core = self.executor.open_position(
            asset,
            entry_price,
            core_notional,
            enriched_row,
            stop_price=enriched_row["stop_price"],
            leg="core",
        )
        self.free_capital -= core_notional
        pos_core.metadata["initial_stop"] = enriched_row["stop_price"]
        pos_core.metadata["p_bos_cont"] = enriched_row.get("p_bos_cont", enriched_row.get("prob_bos_cont", 0.0))
        pos_core.metadata["tier"] = tier_result.get("tier")
        pos_core.metadata["reason"] = tier_result.get("reason")
        self.open_positions[pos_core.trade_id] = pos_core
        self._register_exposure(pos_core)

        if runner_notional > 0:
            pos_runner = self.executor.open_position(
                asset,
                entry_price,
                runner_notional,
                enriched_row,
                stop_price=enriched_row["stop_price"],
                leg="runner",
            )
            self.free_capital -= runner_notional
            pos_runner.metadata["initial_stop"] = enriched_row["stop_price"]
            pos_runner.metadata["p_bos_cont"] = enriched_row.get("p_bos_cont", enriched_row.get("prob_bos_cont", 0.0))
            pos_runner.metadata["tier"] = tier_result.get("tier")
            pos_runner.metadata["reason"] = tier_result.get("reason")
            self.open_positions[pos_runner.trade_id] = pos_runner
            self._register_exposure(pos_runner)

        # Reasoning logging
        reason = self.reason.build(asset, enriched_row, conf, evr, tier_result.get("tier"), risk, hedge_ratio)
        reason["decision_path"] = {
            "gates": gate,
            "danger": danger,
            "cooling_gate": cooling_gate,
            "tiering": tier_result,
            "capital": {
                "ticket_usd": float(capital_out.get("ticket_usd", 0.0) or 0.0),
                "risk_mode": risk,
                "hedge_ratio": hedge_ratio,
                "lock_fraction": float(capital_out.get("lock_fraction", 0.0) or 0.0),
            },
            "position_sizer": {
                "value_usd": float(sizing.get("value", 0.0) or 0.0),
                "qty": float(sizing.get("qty", 0.0) or 0.0),
                "risk_dollars": float(sizing.get("risk_dollars", 0.0) or 0.0),
                "final_notional_usd": float(pos_usd),
            },
        }
        reason["timestamp"] = dt
        self.trade_log.append({"open": pos_core, "reason": reason})

        self.entry_reason_by_trade[pos_core.trade_id] = dict(reason)
        runner_reason = None
        if runner_notional > 0:
            runner_reason = {**reason, "leg": "runner"}
            self.entry_reason_by_trade[pos_runner.trade_id] = dict(runner_reason)
        if self.dashboard:
            self.dashboard.log_event("entry", pos_core.trade_id, reason)
            if runner_reason is not None:
                self.dashboard.log_event("entry", pos_runner.trade_id, runner_reason)

        self._update_equity(dt, enriched_row)

    # ----------------------------------------------------------------------
    def _inject_model_probs(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs specialist/meta/hazard inference on the current feature row and attaches probabilities.
        """
        enriched = dict(row)
        try:
            specialist_list = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
            preds = self.predictor.predict_single(row, specialist_list)
            # flatten preds into row
            for k, v in preds.items():
                if k == "hazard_curve" and isinstance(v, dict):
                    enriched[k] = v
                    continue
                enriched[k] = v
                if k.startswith("prob_"):
                    enriched[f"p_{k[5:]}"] = v
        except Exception as e:
            LOG.warning(f"[ForwardEngine] predictor failed: {e}")
        return enriched

    # ----------------------------------------------------------------------
    def _apply_profit_ladder(self, asset, row, dt):
        to_close = []
        for tid, pos in list(self.open_positions.items()):
            if pos.asset != asset:
                continue
            decision = self.profit_ladder.evaluate(pos, row)
            if decision["new_stop"] is not None:
                pos.stop_price = decision["new_stop"]
            pos.r_mult = float(decision.get("r_now", pos.r_mult or 0.0))
            pos.highest_r = max(float(getattr(pos, "highest_r", 0.0) or 0.0), pos.r_mult)
            if decision["exit"]:
                res = self.executor.exit_position_at(pos, decision["exit_price"], dt)
                to_close.append((tid, pos, res, decision["reason"]))

        for tid, pos, res, reason in to_close:
            self._finalize_exit(tid, pos, res, reason, source="profit_ladder")
        return bool(to_close)

    # ----------------------------------------------------------------------
    # HAZARD TRAILING FOR OPEN POSITIONS
    # ----------------------------------------------------------------------
    def _apply_hazard(self, asset, row, dt):
        if "hazard" not in self.models:
            return False

        to_close = []
        price = row["close"]
        hazard_score = row.get("hazard_score")

        # If hazard_score not in row, compute via predictor using numeric features
        if hazard_score is None:
            try:
                preds = self.predictor.predict_single(row, [])
                hazard_score = preds.get("hazard_score", 0.0)
            except Exception:
                hazard_score = 0.0

        for tid, pos in list(self.open_positions.items()):
            if pos.asset != asset:
                continue

            # bars in trade approx by elapsed time / 15m
            bars_in_trade = int(max(1, (dt - pos.opened_at).total_seconds() // (15 * 60)))
            trail_row = dict(row)
            trail_row["confluence_score"] = pos.metadata.get("conf", trail_row.get("confluence_score", 0.0))
            trail_row["median_r"] = pos.metadata.get("median_r", trail_row.get("median_r", 0.0))

            decision = self.hazard.evaluate(
                row=pd.Series(trail_row),
                hazard=float(hazard_score or 0.0),
                side=getattr(pos, "side", "long"),
                bars_in_trade=bars_in_trade,
                current_stop=getattr(pos, "stop_price", price),
                config=self.cfg_loader.load(),
                leg=getattr(pos, "leg", "core")
            )

            action = decision["action"]
            new_stop = decision["new_stop"]

            if action in ("tighten", "partial"):
                pos.stop_price = new_stop
            if action == "exit":
                res = self.executor.exit_position_at(pos, price, dt)
                to_close.append((tid, pos, res, action))

        for tid, pos, res, action in to_close:
            self._finalize_exit(tid, pos, res, f"hazard_{action}", source="hazard")
        return bool(to_close)

    # ----------------------------------------------------------------------
    # EQUITY UPDATE + COOLING CHECK
    # ----------------------------------------------------------------------
    def _update_equity(self, dt, row=None):
        mtm = 0.0
        for pos in self.open_positions.values():
            mark_price = self.last_prices.get(pos.asset, getattr(pos, "entry_price", 0.0))
            mtm += self.executor.mark_to_market(pos, mark_price)
        self.equity = self.locked_profit + self.free_capital + mtm

        self.max_equity = max(self.max_equity, self.equity)

        dd = (self.max_equity - self.equity) / max(self.max_equity, 1e-9)
        self.current_drawdown = dd
        self.max_drawdown_seen = max(self.max_drawdown_seen, dd)
        if (not self.compound_cooling.enabled) and dd >= abs(self.exec_cfg.get("cooling_dd_trigger", 0.0)):
            self.cooling_until = dt + timedelta(minutes=self.exec_cfg.get("cooling_minutes", 0))
        self.exposure.snapshot(dt, self.equity)

        if self.dashboard:
            self.dashboard.update_state({
                "timestamp": dt,
                "equity": self.equity,
                "free_capital": self.free_capital,
                "locked_profit": self.locked_profit,
                "max_drawdown": self.max_drawdown_seen,
                "open_positions": len(self.open_positions),
                "open_trades": self.open_positions,
                "closed_trades": self.closed_trades,
                "cooling_to": self.cooling_until.isoformat() if self.cooling_until is not None else None,
                "hazard": row.get("hazard_score", row.get("hazard")) if row else None,
                "flow_1h": row.get("p_flow_1h", row.get("prob_flow_1h")) if row else None,
                "confluence": row.get("confluence_score", row.get("conf_score")) if row else None,
                "evr": row.get("evr") if row else None,
                "risk_mode": self.current_risk_mode,
                "hedge_ratio": self.current_hedge_ratio,
                "exposures": self.exposure.current_exposures(self.equity),
            })

    # ----------------------------------------------------------------------
    # LOCK PROFIT INTO VAULT
    # ----------------------------------------------------------------------
    def _lock_profit(self, frac):
        delta = (self.equity - self.locked_profit) * frac
        if delta > 0:
            self._lock_profit_amount(delta)

    def _lock_profit_amount(self, amount):
        delta = min(max(amount, 0.0), self.free_capital)
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
        median_r = float(evr.get("median_r", evr.get("median_R", 0.0)) or 0.0)
        p_bos_cont = float(row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)) or 0.0)

        if conf >= 0.92 and median_r >= 6.0:
            return True

        if evr.get("evr", 0.0) >= 3.0 and p_bos_cont >= 0.40:
            return True

        return False

    def _emit_manual_alert(
        self,
        *,
        asset: str,
        dt,
        row: Dict[str, Any],
        conf: float,
        evr: Dict[str, Any],
        tier_result: Dict[str, Any],
        gate: Dict[str, Any],
        danger: Dict[str, Any],
        cooling_gate: Dict[str, Any],
    ) -> None:
        alert_id = f"{asset}-{pd.Timestamp(dt).strftime('%Y%m%d%H%M')}-alert"
        reason = self.reason.build(
            asset,
            row,
            conf,
            evr,
            tier_result.get("tier"),
            self.current_risk_mode,
            self.current_hedge_ratio,
        )
        reason["decision_path"] = {
            "gates": gate,
            "danger": danger,
            "cooling_gate": cooling_gate,
            "tiering": tier_result,
            "manual_alert_only": True,
        }
        reason["timestamp"] = dt
        if self.dashboard:
            self.dashboard.log_event("alert", alert_id, reason)

    def run_rows(self, asset: str, rows: pd.DataFrame):
        ordered = rows.sort_values("dt").reset_index(drop=True)
        for _, row in ordered.iterrows():
            self.on_bar(asset, row.to_dict())
        return self.state_snapshot()

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": self.last_timestamp,
            "starting_capital": self.exec_cfg.get("starting_equity", 0.0),
            "equity": self.equity,
            "free_capital": self.free_capital,
            "locked_profit": self.locked_profit,
            "max_drawdown": self.max_drawdown_seen,
            "risk_mode": self.current_risk_mode,
            "hedge_ratio": self.current_hedge_ratio,
            "cooling_to": self.cooling_until.isoformat() if self.cooling_until is not None else None,
            "open_trades": self.open_positions,
            "closed_trades": self.closed_trades,
            "exposures": self.exposure.current_exposures(self.equity),
            "manual_alert_only": self.manual_alert_only,
        }

    def _register_exposure(self, pos) -> None:
        if getattr(pos, "side", "long") == "short":
            self.exposure.register_short(float(pos.notional_usd), asset=pos.asset)
        else:
            self.exposure.register_long(float(pos.notional_usd), asset=pos.asset)

    def _release_exposure(self, pos) -> None:
        if getattr(pos, "side", "long") == "short":
            self.exposure.release(pos.asset, short_notional=float(pos.notional_usd))
        else:
            self.exposure.release(pos.asset, long_notional=float(pos.notional_usd))

    def _finalize_exit(
        self,
        trade_id: str,
        pos,
        res: Dict[str, Any],
        reason: str,
        *,
        source: str,
    ) -> None:
        self.free_capital += float(res["value"])
        self._release_exposure(pos)
        payload = {
            **res,
            "trade_id": trade_id,
            "asset": pos.asset,
            "side": pos.side,
            "leg": pos.leg,
            "entry_price": pos.entry_price,
            "entry_ts": pos.opened_at,
            "stop_price": pos.stop_price,
            "conf": pos.metadata.get("conf"),
            "evr": pos.metadata.get("evr"),
            "tier": pos.metadata.get("tier"),
            "reason": reason,
            "source": source,
            "timestamp": res.get("exit_ts"),
            "bars_in_trade": getattr(pos, "bars_in_trade", None),
            "highest_r": getattr(pos, "highest_r", None),
            "r_mult_at_exit": getattr(pos, "r_mult", None),
        }
        entry_reason = self.entry_reason_by_trade.pop(trade_id, None)
        if entry_reason:
            payload["entry_reasoning"] = entry_reason
        self.closed_trades[trade_id] = payload
        self.open_positions.pop(trade_id, None)
        if self.dashboard:
            self.dashboard.log_event("exit", trade_id, payload)
