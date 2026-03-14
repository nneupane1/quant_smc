"""Live trading orchestrator aligned to the forward-test execution contract."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.gating.profit_ladder import ProfitLadderManager
from quant_system.execution.gating.tiering import TieringEngine
from quant_system.execution.risk.capital_allocator import CapitalAllocator
from quant_system.execution.risk.compound_cooling import CompoundCoolingPolicy
from quant_system.execution.risk.exposure_tracker import ExposureTracker
from quant_system.execution.risk.mpc_risk import MPCRiskManager
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.forward_test.forward_reasoning_attach import ReasoningAttach
from quant_system.live.kraken_live_client import KrakenLiveClient
from quant_system.live.live_executor import LiveExecutor
from quant_system.live_data.live_feature_enricher import LiveFeatureEnricher
from quant_system.live_data.quote_state import QuoteState
from quant_system.live_data.tf_builder import TFBuilder
from quant_system.ml.predict.model_predictor import ModelPredictor, resolve_inference_preference
from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.utils.logger import get_logger

LOG = get_logger("live_orchestrator")


class LiveOrchestrator:
    def __init__(self, config_loader: ConfigLoader, registry: ModelRegistry, dashboard_adapter=None):
        self.cfg_loader = config_loader
        self.cfg = config_loader.load()
        self.exec_cfg = self.cfg.get("execution", {})
        pref_cfg = self.cfg.get("inference_preference", {}) if isinstance(self.cfg.get("inference_preference", {}), dict) else {}
        pref = resolve_inference_preference(pref_cfg)
        self.routing_mode = str(pref["routing_mode"])
        self.challenger_mode = str(pref["challenger_mode"])
        self.allow_hybrid_explicit = bool(pref["allow_hybrid_explicit"])
        self.manual_alert_only = bool(self.exec_cfg.get("manual_alert_only", False))
        self.registry = registry
        self.dashboard = dashboard_adapter

        self.confluence = ConfluenceEngine(config_loader)
        self.evr = EVRCalculator(config_loader)
        self.gates = GateEvaluator(config_loader)
        self.tiering = TieringEngine(config_loader)
        self.hazard = HazardTrailingEngine(config_loader)
        self.profit_ladder = ProfitLadderManager(config_loader)
        self.mpc = MPCRiskManager(config_loader)
        self.capital = CapitalAllocator(config_loader)
        self.compound_cooling = CompoundCoolingPolicy(config_loader)
        self.position_sizer = PositionSizer(config_loader)
        self.exposure = ExposureTracker(self.cfg)
        self.reason = ReasoningAttach()

        self.predictor = ModelPredictor(
            registry,
            routing_mode=self.routing_mode,
            challenger_mode=self.challenger_mode,
            allow_hybrid_explicit=self.allow_hybrid_explicit,
        )
        self.models_loaded = False

        self.feed = KrakenLiveClient(config_loader)
        self.executor = LiveExecutor(config_loader, dashboard_adapter=dashboard_adapter)
        self.assets_cfg = self.cfg.get("assets", {})
        self.assets_meta = self.assets_cfg.get("metadata", {})
        self.default_asset = self.cfg.get("default_asset") or (next(iter(self.assets_meta)) if self.assets_meta else "BTCUSD")
        self.leverage_cfg = self.cfg.get("live_trading", {}).get("leverage", {})
        assets = list(self.assets_meta.keys()) or [self.default_asset]
        self.quote_state = {asset: QuoteState() for asset in assets}
        self.tf_builder = {asset: TFBuilder() for asset in assets}
        self.live_enricher = LiveFeatureEnricher(config_loader)
        self.strict_gate_mode = bool(self.exec_cfg.get("gates", {}).get("strict_mode", False))
        self.pair_aliases = self._build_pair_aliases()

        self.buffers = {"1m": []}
        self.bar_index = 0
        self.cooling_until = None
        self.current_drawdown = 0.0
        self.max_drawdown_seen = 0.0
        self._max_equity = float(self.exec_cfg.get("starting_equity", 0.0))
        self.current_risk_mode = None
        self.current_hedge_ratio = 0.0
        self.closed_trades: Dict[str, Dict[str, Any]] = {}
        self.last_prices: Dict[str, float] = {}
        self.last_timestamp = None
        self.entry_reason_by_trade: Dict[str, Dict[str, Any]] = {}

        LOG.info("[LiveOrchestrator] Initialized.")

    def load_models(self):
        self.models_loaded = True
        resolved_specialists = self.predictor.warmup_specialists(
            ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
        )
        resolved_stacks = self.predictor.warmup_stacks(["meta_model", "confluence_model"])
        specialist_bundle = self.predictor.specialist_bundle_map()
        stack_bundle = self.predictor.stack_bundle_map()
        route_status = self.predictor.routing_status()
        LOG.info(
            "[LiveOrchestrator] Inference routing requested=%s effective=%s challenger=%s "
            "allow_hybrid=%s note=%s specialists=%s stacks=%s",
            route_status["requested_mode"],
            route_status["effective_mode"],
            route_status["challenger_mode"],
            route_status["allow_hybrid_explicit"],
            route_status["note"] or "ready",
            resolved_specialists,
            resolved_stacks,
        )
        if self.dashboard:
            self.dashboard.log_event(
                "models_loaded",
                None,
                {
                    "version": "latest",
                    "routing_mode_requested": route_status["requested_mode"],
                    "challenger_mode": route_status["challenger_mode"],
                    "allow_hybrid_explicit": route_status["allow_hybrid_explicit"],
                    "routing_note": route_status["note"],
                    "inference_source_mode": self.predictor.source_mode(),
                    "specialist_model_source": resolved_specialists,
                    "stack_model_source": resolved_stacks,
                    "specialist_model_bundle": specialist_bundle,
                    "stack_model_bundle": stack_bundle,
                },
            )

    def run(self):
        LOG.info("[LiveOrchestrator] Live loop started.")
        if not self.models_loaded:
            self.load_models()

        for c1 in self.feed.run_stream():
            if c1 is None:
                continue
            candle_1m = dict(c1)
            asset = self._resolve_asset(str(candle_1m.get("asset", ""))) or self.default_asset
            if asset not in self.quote_state:
                self.quote_state[asset] = QuoteState()
            if asset not in self.tf_builder:
                self.tf_builder[asset] = TFBuilder()
            candle_1m["asset"] = asset
            if "dt" not in candle_1m:
                candle_1m["dt"] = datetime.utcfromtimestamp(int(candle_1m["timestamp"]))

            self.quote_state[asset].push_1m(candle_1m)
            emits = self.tf_builder[asset].push_1m(candle_1m)
            for tf, bar in emits.items():
                bar["asset"] = asset
                self.quote_state[asset].push_tf(tf, bar)
                if tf != "15m":
                    continue
                enriched = self.live_enricher.enrich(self.quote_state[asset], asset, bar)
                if enriched is None and self.strict_gate_mode:
                    LOG.info("[LiveOrchestrator] strict_mode drop %s 15m bar due to missing HTF context", asset)
                    continue
                self.on_bar(asset, enriched or bar)

    def run_rows(self, asset: str, rows: pd.DataFrame):
        if not self.models_loaded:
            self.load_models()
        ordered = rows.sort_values("dt").reset_index(drop=True)
        for _, row in ordered.iterrows():
            self.on_bar(asset, row.to_dict())
        return self.state_snapshot()

    def on_bar(self, asset: str, row: Dict[str, Any]):
        if not self.models_loaded:
            self.load_models()

        dt = pd.to_datetime(row.get("dt"))
        self.last_timestamp = dt
        self.last_prices[asset] = float(row["close"])
        self.bar_index += 1

        if self.dashboard:
            self.dashboard.update_candles({asset: row})

        series = pd.Series(row).copy()
        series["dt"] = dt
        series["asset"] = asset
        enriched = self._inject_model_probs(series)

        had_ladder_exit = self._apply_profit_ladder(asset, enriched, dt)
        had_hazard_exit = self._apply_hazard(asset, enriched, dt)
        self._refresh_equity(dt, enriched)
        if had_ladder_exit or had_hazard_exit:
            return

        danger = self.compound_cooling.evaluate_danger(
            dt=dt,
            equity=self.executor.equity,
            free_capital=self.executor.free_capital,
            locked_profit=self.executor.locked_profit,
            drawdown=self.current_drawdown,
            row=enriched.to_dict(),
            cooling_until=self.cooling_until,
        )
        if danger["lock_amount"] > 0:
            lock_delta = min(danger["lock_amount"], self.executor.free_capital)
            self.executor.locked_profit += lock_delta
            self.executor.free_capital -= lock_delta
        if danger["cooling_until"] is not None:
            self.cooling_until = danger["cooling_until"]

        gate = self.gates.evaluate(enriched, enriched.get("side", "long"))
        if not gate["passed"]:
            self._refresh_equity(dt, enriched)
            return

        conf_result = self.confluence.compute(self.models_loaded, enriched)
        conf = conf_result.get("conf_score", conf_result.get("confluence_score", 0.0))
        enriched["confluence_score"] = conf
        enriched["conf_score"] = conf

        side = enriched.get("side", "long")
        evr_pack = self.evr.compute(self.models_loaded, enriched, side=side)
        enriched["evr"] = evr_pack.get("evr")
        enriched["median_r"] = evr_pack.get("median_r")
        enriched["stop_price"] = evr_pack.get("stop_price", enriched["close"])

        tier = self.tiering.classify(
            row=enriched,
            confluence_pass=bool(conf_result.get("passed", False)),
            evr_result=evr_pack,
            hazard_score=float(danger["metrics"]["hazard"] or 0.0),
            bar_index=self.bar_index,
        )
        if self._moonshot_override(enriched, conf, evr_pack):
            tier = {"tier": "A+", "execute": True, "reason": "moonshot_override"}

        cooling_gate = self.compound_cooling.allow_entry(
            now=dt,
            cooling_until=self.cooling_until,
            conf=conf,
            evr=evr_pack,
            hazard=danger["metrics"]["hazard"],
        )
        if not cooling_gate["allow"]:
            self._refresh_equity(dt, enriched)
            return

        if tier.get("tier") == "skip" or not tier.get("execute", False):
            self._refresh_equity(dt, enriched)
            return

        if self.manual_alert_only:
            self._emit_manual_alert(
                asset=asset,
                dt=dt,
                row=enriched,
                conf=float(conf),
                evr_pack=evr_pack,
                tier=tier,
                gate=gate,
                danger=danger,
                cooling_gate=cooling_gate,
            )
            self._refresh_equity(dt, enriched)
            return

        capital_out = self.capital.allocate(
            equity=self.executor.equity,
            free_capital=self.executor.free_capital,
            locked_profit=self.executor.locked_profit,
            row=enriched.to_dict(),
            mpc_manager=self.mpc,
        )
        risk_mode = capital_out["risk_mode"]
        if risk_mode is None:
            risk_mode = float(np.clip(float(capital_out.get("ticket_usd", 0.0)) / max(self.executor.equity, 1e-9), 0.0, 1.0))
        self.current_risk_mode = risk_mode
        self.current_hedge_ratio = capital_out["hedge_ratio"]

        if capital_out["lock_fraction"] > 0:
            lock_delta = (self.executor.equity - self.executor.locked_profit) * capital_out["lock_fraction"]
            lock_delta = max(min(lock_delta, self.executor.free_capital), 0.0)
            self.executor.locked_profit += lock_delta
            self.executor.free_capital -= lock_delta

        sizing = self.position_sizer.size_position(
            row=enriched,
            equity=max(self.executor.equity, 0.0),
            side=side,
            stop_price=float(enriched.get("stop_price", enriched["close"])),
            risk_mode=float(risk_mode),
            hedge_ratio=float(capital_out.get("hedge_ratio", 0.0)),
        )
        pos_usd = min(
            float(capital_out["ticket_usd"]),
            float(sizing.get("value", 0.0)),
            float(self.executor.free_capital),
        )
        enriched["sizer_value_usd"] = float(sizing.get("value", 0.0))
        enriched["sizer_qty"] = float(sizing.get("qty", 0.0))
        enriched["sizer_risk_dollars"] = float(sizing.get("risk_dollars", 0.0))
        enriched["position_notional_usd"] = float(pos_usd)
        if pos_usd <= 0 or pos_usd > self.executor.free_capital:
            self._refresh_equity(dt, enriched)
            return

        split = self.exec_cfg.get("runner_split", {})
        core_frac = float(split.get("core_frac", 0.7))
        runner_frac = 1.0 - core_frac
        entry = float(enriched["close"])
        leverage_plan = self._resolve_entry_leverage(
            asset=asset,
            row=enriched,
            tier=tier,
            conf=float(conf),
            evr_pack=evr_pack,
            hazard=float(danger["metrics"]["hazard"] or 0.0),
        )
        entry_leverage = int(leverage_plan["leverage"])

        def _meta(leg: str) -> Dict[str, Any]:
            return {
                "leg": leg,
                "conf": conf,
                "tier": tier.get("tier"),
                "direction": side,
                "median_r": evr_pack.get("median_r"),
                "evr": evr_pack.get("evr"),
                "p_bos_cont": enriched.get("p_bos_cont", enriched.get("prob_bos_cont", 0.0)),
                "initial_stop": enriched["stop_price"],
                "reason": tier.get("reason"),
                "leverage": entry_leverage,
                "leverage_reason": leverage_plan["reason"],
            }

        pos_core = None
        core_usd = pos_usd * core_frac
        if core_usd > 0:
            trade_id = f"{asset}-{uuid.uuid4().hex[:8]}-core"
            pos_core = self.executor.open_position(trade_id, asset, core_usd, entry, _meta("core"), stop_price=enriched["stop_price"])
            if pos_core:
                self._register_exposure(pos_core)

        pos_runner = None
        runner_usd = pos_usd * runner_frac
        if runner_usd > 0:
            trade_id = f"{asset}-{uuid.uuid4().hex[:8]}-runner"
            pos_runner = self.executor.open_position(trade_id, asset, runner_usd, entry, _meta("runner"), stop_price=enriched["stop_price"])
            if pos_runner:
                self._register_exposure(pos_runner)

        reason = self.reason.build(asset, enriched.to_dict(), conf, evr_pack, tier.get("tier"), self.current_risk_mode, self.current_hedge_ratio)
        reason["decision_path"] = {
            "gates": gate,
            "danger": danger,
            "cooling_gate": cooling_gate,
            "tiering": tier,
            "capital": {
                "ticket_usd": float(capital_out.get("ticket_usd", 0.0) or 0.0),
                "risk_mode": risk_mode,
                "hedge_ratio": self.current_hedge_ratio,
                "lock_fraction": float(capital_out.get("lock_fraction", 0.0) or 0.0),
                "session_ticket_multiplier": float(capital_out.get("session_ticket_multiplier", 1.0) or 1.0),
            },
            "position_sizer": {
                "value_usd": float(sizing.get("value", 0.0) or 0.0),
                "qty": float(sizing.get("qty", 0.0) or 0.0),
                "risk_dollars": float(sizing.get("risk_dollars", 0.0) or 0.0),
                "final_notional_usd": float(pos_usd),
            },
            "leverage": leverage_plan,
        }
        reason["timestamp"] = dt
        if pos_core:
            self.entry_reason_by_trade[pos_core.trade_id] = dict(reason)
        runner_reason = None
        if pos_runner:
            runner_reason = {**reason, "leg": "runner"}
            self.entry_reason_by_trade[pos_runner.trade_id] = dict(runner_reason)
        if self.dashboard and pos_core:
            self.dashboard.log_event("entry", pos_core.trade_id, reason)
            if pos_runner and runner_reason is not None:
                self.dashboard.log_event("entry", pos_runner.trade_id, runner_reason)

        self._refresh_equity(dt, enriched)

    def _apply_profit_ladder(self, asset: str, row: pd.Series, dt) -> bool:
        to_close = []
        for tid, pos in list(self.executor.positions.items()):
            if pos.asset != asset:
                continue
            decision = self.profit_ladder.evaluate(pos, row)
            if decision["new_stop"] is not None:
                pos.stop_price = decision["new_stop"]
            pos.r_mult = float(decision.get("r_now", pos.r_mult or 0.0))
            pos.highest_r = max(float(getattr(pos, "highest_r", 0.0) or 0.0), pos.r_mult)
            if decision["exit"]:
                res = self.executor.exit_position_at(tid, decision["exit_price"], dt)
                if res is not None:
                    to_close.append((tid, pos, res, decision["reason"]))

        for tid, pos, res, reason in to_close:
            self._finalize_exit(tid, pos, res, reason, source="profit_ladder")
        return bool(to_close)

    def _apply_hazard(self, asset: str, row: pd.Series, dt) -> bool:
        to_close = []
        for tid, pos in list(self.executor.positions.items()):
            if pos.asset != asset:
                continue
            hazard_val = row.get("hazard_score", row.get("hazard", 0.0))
            trail_row = row.copy()
            trail_row["confluence_score"] = pos.meta.get("conf", trail_row.get("confluence_score", 0.0))
            trail_row["median_r"] = pos.meta.get("median_r", trail_row.get("median_r", 0.0))
            trail = self.hazard.evaluate(
                row=trail_row,
                hazard=float(hazard_val or 0.0),
                side=pos.side,
                bars_in_trade=pos.bars_in_trade,
                current_stop=pos.stop_price or row["close"],
                config=self.cfg_loader,
                leg=pos.meta.get("leg", "core"),
            )
            pos.bars_in_trade += 1
            if trail["action"] in ("partial", "tighten"):
                pos.stop_price = trail["new_stop"]
            if trail["action"] == "exit":
                res = self.executor.exit_position_at(tid, trail["new_stop"], dt)
                if res is not None:
                    to_close.append((tid, pos, res, f"hazard_{trail['action']}"))

        for tid, pos, res, reason in to_close:
            self._finalize_exit(tid, pos, res, reason, source="hazard")
        return bool(to_close)

    def _refresh_equity(self, now, row: Optional[pd.Series] = None):
        if row is not None:
            asset = row.get("asset")
            if asset is not None:
                self.last_prices[str(asset)] = float(row.get("close", self.last_prices.get(str(asset), 0.0)))
        self.executor.refresh_equity(self.last_prices)
        self._max_equity = max(self._max_equity, self.executor.equity)
        dd = (self._max_equity - self.executor.equity) / max(self._max_equity, 1e-9)
        self.current_drawdown = dd
        self.max_drawdown_seen = max(self.max_drawdown_seen, dd)
        if (not self.compound_cooling.enabled) and dd >= self.exec_cfg.get("cooling_dd_trigger", 1.0):
            minutes = self.exec_cfg.get("cooling_minutes", 0)
            self.cooling_until = now + pd.Timedelta(minutes=minutes)
        self.exposure.snapshot(now, self.executor.equity)
        self._update_dashboard(row if row is not None else pd.Series({"dt": now}), 0.0 if row is None else row.get("confluence_score", row.get("conf_score")), {"evr": None if row is None else row.get("evr")})

    def _ready_bar(self) -> bool:
        if len(self.buffers["1m"]) < 15:
            return False
        return self.buffers["1m"][-1]["timestamp"] % 900 == 0

    def _agg_15m(self) -> Optional[Dict[str, Any]]:
        rows = self.buffers["1m"][-15:]
        if len(rows) < 15:
            return None
        return {
            "timestamp": rows[-1]["timestamp"],
            "dt": datetime.utcfromtimestamp(rows[-1]["timestamp"]),
            "open": rows[0]["open"],
            "high": max(r["high"] for r in rows),
            "low": min(r["low"] for r in rows),
            "close": rows[-1]["close"],
            "volume": sum(r["volume"] for r in rows),
            "asset": rows[-1].get("asset", "BTCUSD"),
        }

    def _inject_model_probs(self, row: pd.Series) -> pd.Series:
        enriched = row.copy()
        try:
            specialist_list = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
            preds = self.predictor.predict_single(row, specialist_list)
            for k, v in preds.items():
                if k == "hazard_curve" and isinstance(v, dict):
                    enriched[k] = v
                    continue
                enriched[k] = v
                if k.startswith("prob_"):
                    enriched[f"p_{k[5:]}"] = v
            if "hazard_score" in preds:
                enriched["hazard"] = preds["hazard_score"]
        except Exception as exc:
            LOG.warning("[LiveOrchestrator] predictor failed: %s", exc)
        return enriched

    def _build_pair_aliases(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for asset, meta in self.assets_meta.items():
            variants = {
                asset,
                meta.get("symbol", ""),
                meta.get("kraken_pair", ""),
                f"{meta.get('base', '')}/{meta.get('quote', '')}" if meta.get("base") and meta.get("quote") else "",
                f"X{meta.get('base', '')}Z{meta.get('quote', '')}" if meta.get("base") and meta.get("quote") else "",
                f"XXBTZUSD" if meta.get("kraken_pair", "").startswith("XBT") and meta.get("quote") == "USD" else "",
            }
            for v in variants:
                if v:
                    mapping[v.upper()] = asset
                    mapping[v.replace("/", "").upper()] = asset
        return mapping

    def _resolve_asset(self, pair: str) -> Optional[str]:
        if not pair:
            return None
        return self.pair_aliases.get(pair.upper()) or self.pair_aliases.get(pair.replace("/", "").upper())

    def _update_dashboard(self, row: pd.Series, conf: float, evr: Dict[str, Any]):
        if not self.dashboard:
            return
        self.dashboard.update_state(
            {
                "timestamp": row.get("dt", self.last_timestamp),
                "equity": self.executor.equity,
                "free_capital": self.executor.free_capital,
                "locked_profit": self.executor.locked_profit,
                "confluence": conf,
                "evr": evr.get("evr") if isinstance(evr, dict) else evr,
                "hazard": row.get("hazard_score", row.get("hazard")),
                "flow_1h": row.get("p_flow_1h", row.get("prob_flow_1h")),
                "open_positions": len(self.executor.positions),
                "open_trades": self.executor.positions,
                "closed_trades": self.closed_trades,
                "cooling_to": self.cooling_until.isoformat() if self.cooling_until is not None else None,
                "max_drawdown": self.max_drawdown_seen,
                "risk_mode": self.current_risk_mode,
                "hedge_ratio": self.current_hedge_ratio,
                "exposures": self.exposure.current_exposures(self.executor.equity),
            }
        )

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

    def _finalize_exit(self, trade_id: str, pos, res: Dict[str, Any], reason: str, *, source: str) -> None:
        self._release_exposure(pos)
        payload = {
            **res,
            "trade_id": trade_id,
            "asset": pos.asset,
            "side": pos.side,
            "leg": pos.meta.get("leg"),
            "entry_price": pos.entry_price,
            "entry_ts": pos.opened_at,
            "stop_price": pos.stop_price,
            "conf": pos.meta.get("conf"),
            "evr": pos.meta.get("evr"),
            "tier": pos.meta.get("tier"),
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
        if self.dashboard:
            self.dashboard.log_event("exit", trade_id, payload)

    def _moonshot_override(self, row: pd.Series, conf: float, evr: Dict[str, Any]) -> bool:
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
        row: pd.Series,
        conf: float,
        evr_pack: Dict[str, Any],
        tier: Dict[str, Any],
        gate: Dict[str, Any],
        danger: Dict[str, Any],
        cooling_gate: Dict[str, Any],
    ) -> None:
        alert_id = f"{asset}-{pd.Timestamp(dt).strftime('%Y%m%d%H%M')}-alert"
        reason = self.reason.build(
            asset,
            row.to_dict(),
            conf,
            evr_pack,
            tier.get("tier"),
            self.current_risk_mode,
            self.current_hedge_ratio,
        )
        reason["decision_path"] = {
            "gates": gate,
            "danger": danger,
            "cooling_gate": cooling_gate,
            "tiering": tier,
            "manual_alert_only": True,
        }
        reason["timestamp"] = dt
        if self.dashboard:
            self.dashboard.log_event("alert", alert_id, reason)

    def _resolve_entry_leverage(
        self,
        *,
        asset: str,
        row: pd.Series,
        tier: Dict[str, Any],
        conf: float,
        evr_pack: Dict[str, Any],
        hazard: float,
    ) -> Dict[str, Any]:
        lev_cfg = self.leverage_cfg or {}
        if not bool(lev_cfg.get("enabled", False)):
            return {"leverage": 1, "reason": "leverage_disabled"}

        asset_meta = self.assets_meta.get(asset, {})
        if not bool(asset_meta.get("leverage_allowed", False)):
            return {"leverage": 1, "reason": "asset_not_leverageable"}

        max_lev = int(lev_cfg.get("max", 1))
        min_lev = int(lev_cfg.get("min", 1))
        default_lev = int(np.clip(int(lev_cfg.get("default", 1)), min_lev, max_lev))
        high_conf_lev = int(np.clip(int(lev_cfg.get("high_conf_leverage", 2)), min_lev, max_lev))

        tier_name = str(tier.get("tier", "")).strip()
        allowed_tiers = {str(t).strip() for t in lev_cfg.get("high_conf_tiers", ["A+"])}
        min_conf = float(lev_cfg.get("high_conf_min_conf", 0.82))
        min_evr = float(lev_cfg.get("high_conf_min_evr", 1.8))
        max_hazard = float(lev_cfg.get("high_conf_max_hazard", 0.25))
        min_bos = float(lev_cfg.get("high_conf_min_bos_cont", 0.55))
        p_bos_cont = float(row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)) or 0.0)

        high_conf = (
            tier_name in allowed_tiers
            and conf >= min_conf
            and float(evr_pack.get("evr", 0.0) or 0.0) >= min_evr
            and hazard <= max_hazard
            and p_bos_cont >= min_bos
        )
        if not high_conf:
            return {"leverage": default_lev, "reason": "default_profile"}

        return {
            "leverage": high_conf_lev,
            "reason": "high_confidence",
            "tier": tier_name,
            "conf": conf,
            "evr": float(evr_pack.get("evr", 0.0) or 0.0),
            "hazard": hazard,
            "p_bos_cont": p_bos_cont,
        }

    def state_snapshot(self) -> Dict[str, Any]:
        source_mode = self.predictor.source_mode() if self.predictor is not None else "unknown"
        specialist_source = self.predictor.specialist_source_map() if self.predictor is not None else {}
        stack_source = self.predictor.stack_source_map() if self.predictor is not None else {}
        specialist_bundle = self.predictor.specialist_bundle_map() if self.predictor is not None else {}
        stack_bundle = self.predictor.stack_bundle_map() if self.predictor is not None else {}
        route_status = self.predictor.routing_status() if self.predictor is not None else {}
        return {
            "timestamp": self.last_timestamp,
            "starting_capital": self.exec_cfg.get("starting_equity", 0.0),
            "equity": self.executor.equity,
            "free_capital": self.executor.free_capital,
            "locked_profit": self.executor.locked_profit,
            "max_drawdown": self.max_drawdown_seen,
            "risk_mode": self.current_risk_mode,
            "hedge_ratio": self.current_hedge_ratio,
            "cooling_to": self.cooling_until.isoformat() if self.cooling_until is not None else None,
            "open_trades": self.executor.positions,
            "closed_trades": self.closed_trades,
            "exposures": self.exposure.current_exposures(self.executor.equity),
            "manual_alert_only": self.manual_alert_only,
            "prefer_tcn_specialists": source_mode == "tcn",
            "routing_mode_requested": route_status.get("requested_mode", self.routing_mode),
            "challenger_mode": route_status.get("challenger_mode", self.challenger_mode),
            "active_slot": route_status.get("active_slot", "production"),
            "allow_hybrid_explicit": route_status.get("allow_hybrid_explicit", self.allow_hybrid_explicit),
            "routing_note": route_status.get("note", ""),
            "inference_source_mode": source_mode,
            "specialist_model_source": specialist_source,
            "stack_model_source": stack_source,
            "specialist_model_bundle": specialist_bundle,
            "stack_model_bundle": stack_bundle,
        }
