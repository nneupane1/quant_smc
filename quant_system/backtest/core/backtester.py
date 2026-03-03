"""
Backtester
Multi-asset backtesting loop that wires feature rows through confluence/EVR/tiering,
opens positions via ExecutionSimulator, trails/exit via hazard logic, and logs trades.
"""

from typing import Dict, Any, Optional

import pandas as pd

from quant_system.backtest.core.execution_simulator import ExecutionSimulator, Position
from quant_system.backtest.core.trade_log import TradeLog
from quant_system.backtest.core.metrics import BacktestMetrics
from quant_system.backtest.replay.replay_timeline import build_timeline

from quant_system.execution.gating.hazard_trailing import HazardTrailingEngine
from quant_system.execution.risk.mpc_risk import MPCRiskManager
from quant_system.execution.risk.position_sizer import PositionSizer
from quant_system.execution.risk.exposure_tracker import ExposureTracker
from quant_system.execution.risk.capital_allocator import CapitalAllocator
from quant_system.execution.risk.compound_cooling import CompoundCoolingPolicy
from quant_system.execution.gating.tiering import TieringEngine
from quant_system.execution.gating.evr import EVRCalculator
from quant_system.execution.gating.confluence import ConfluenceEngine
from quant_system.execution.gating.gates import GateEvaluator
from quant_system.execution.gating.profit_ladder import ProfitLadderManager

from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.predict.model_predictor import ModelPredictor
from quant_system.config.config_loader import ConfigLoader
from quant_system.config.config_manager import ConfigManager

from quant_system.utils.logger import get_logger

LOG = get_logger("backtester")
import numpy as np


def _load_config(config_loader: Any) -> Dict[str, Any]:
    """
    Accepts ConfigLoader, ConfigManager, or raw dict and returns merged config dict.
    """
    if isinstance(config_loader, ConfigManager):
        return config_loader.full
    if isinstance(config_loader, ConfigLoader):
        return config_loader.load()
    if isinstance(config_loader, dict):
        return config_loader
    raise TypeError("Unsupported config loader type for Backtester.")


class Backtester:
    """
    Multi-asset execution engine for historical evaluation.
    """

    def __init__(
        self,
        config_loader: Any,
        model_registry: ModelRegistry,
        dashboard_adapter=None,
    ):
        self.cfg = _load_config(config_loader)
        self.reg = model_registry
        self.dashboard = dashboard_adapter

        self.assets_cfg = self.cfg.get("assets", {})
        self.exec_cfg = self.cfg.get("execution", {})

        self.simulator = ExecutionSimulator(self.cfg)
        self.tiering = TieringEngine(self.cfg)
        self.evr_calc = EVRCalculator(self.cfg)
        self.confluence = ConfluenceEngine(self.cfg)
        self.gates = GateEvaluator(self.cfg)

        self.hazard_engine = HazardTrailingEngine(self.cfg)
        self.profit_ladder = ProfitLadderManager(self.cfg)
        self.predictor = ModelPredictor(model_registry)
        self.mpc = MPCRiskManager(self.cfg)
        self.capital = CapitalAllocator(self.cfg)
        self.compound_cooling = CompoundCoolingPolicy(self.cfg)
        self.position_sizer = PositionSizer(self.cfg)
        self.exposure = ExposureTracker(self.cfg)

        self.trade_log = TradeLog()

        LOG.info("[Backtester] Initialized with multi-asset support")

    # ----------------------------------------------------------------------
    # RUN BACKTEST
    # ----------------------------------------------------------------------
    def run(self, asset_frames: Dict[str, pd.DataFrame], model_version: Optional[str] = None):
        """
        asset_frames = {
            "BTCUSDT": df_btc_15m,
            "ETHUSDT": df_eth_15m,
            ...
        }
        """

        LOG.info(f"[Backtester] Starting multi-asset backtest using model version={model_version}")

        # Load models if registry provided
        models = {}
        if model_version:
            LOG.info("[Backtester] model_version selection is not wired yet; using registry latest lookups.")

        # Combined chronological index across assets
        timeline = self._merge_timelines(asset_frames)

        equity = float(self.exec_cfg.get("starting_equity", 0))
        free_capital = equity
        locked_profit = 0.0
        max_equity = equity

        cooling_until: Optional[pd.Timestamp] = None
        open_positions: Dict[str, Position] = {}
        equity_history = []

        # Loop through unified timeline
        for bar_index, t in enumerate(timeline):
            # Per-asset evaluation
            for asset, df in asset_frames.items():
                row = df.loc[df["dt"] == t]
                if row.empty:
                    continue

                row = row.iloc[0]

                # Model inference (specialists/meta/hazard)
                row_enriched = self._inject_model_probs(row)

                # Push candle to dashboard
                if self.dashboard:
                    self.dashboard.update_candles({asset: row_enriched})

                # Hazard trailing for open trades on this asset
                to_close = []
                for tid, pos in list(open_positions.items()):
                    if pos.asset != asset:
                        continue

                    ladder = self.profit_ladder.evaluate(pos, row_enriched)
                    if ladder["new_stop"] is not None:
                        pos.stop_price = ladder["new_stop"]
                    if ladder["exit"]:
                        exit_price = ladder["exit_price"]
                        exit_info = self.simulator.exit_position(pos, exit_price, reason=ladder["reason"])
                        free_capital += pos.size_usd + exit_info["pnl"]
                        self.trade_log.append_close(
                            pos,
                            exit_info["pnl"],
                            t,
                            exit_price,
                            reason=ladder["reason"],
                            regime=row_enriched.get("regime_state"),
                        )
                        to_close.append(tid)
                        continue

                    hazard_val = self._estimate_hazard(models.get("hazard"), row_enriched)
                    trail_row = row_enriched.copy()
                    trail_row["confluence_score"] = pos.conf if pos.conf is not None else trail_row.get("confluence_score", 0.0)
                    trail_row["median_r"] = pos.metadata.get("median_r", trail_row.get("median_r", 0.0))
                    trail = self.hazard_engine.evaluate(
                        row=trail_row,
                        hazard=hazard_val,
                        side=pos.side,
                        bars_in_trade=pos.metadata.get("bars", 0),
                        current_stop=pos.stop_price,
                        config=self.cfg,
                        leg=pos.metadata.get("leg", "core"),
                    )

                    if trail["action"] == "exit":
                        exit_price = trail["new_stop"]
                        exit_info = self.simulator.exit_position(pos, exit_price, reason="hazard_exit")
                        free_capital += pos.size_usd + exit_info["pnl"]
                        self.trade_log.append_close(pos, exit_info["pnl"], t, exit_price, reason="hazard_exit", regime=row_enriched.get("regime_state"))
                        to_close.append(tid)
                    elif trail["action"] in ("tighten", "partial"):
                        pos.stop_price = trail["new_stop"]
                        pos.metadata["bars"] = pos.metadata.get("bars", 0) + 1
                    else:
                        pos.metadata["bars"] = pos.metadata.get("bars", 0) + 1

                for tid in to_close:
                    open_positions.pop(tid, None)

                equity = self._portfolio_equity(asset_frames, open_positions, free_capital, locked_profit, t)
                max_equity = max(max_equity, equity)
                current_drawdown = (max_equity - equity) / max(max_equity, 1e-9)
                danger = self.compound_cooling.evaluate_danger(
                    dt=pd.Timestamp(t),
                    equity=equity,
                    free_capital=free_capital,
                    locked_profit=locked_profit,
                    drawdown=current_drawdown,
                    row=row_enriched.to_dict(),
                    cooling_until=cooling_until,
                )
                if danger["lock_amount"] > 0:
                    lock_delta = min(danger["lock_amount"], free_capital)
                    locked_profit += lock_delta
                    free_capital -= lock_delta
                    equity = self._portfolio_equity(asset_frames, open_positions, free_capital, locked_profit, t)
                if danger["cooling_until"] is not None:
                    cooling_until = danger["cooling_until"]

                # Confluence
                conf_result = self._compute_confluence(row_enriched)
                conf_score = conf_result["score"]
                conf_pass = conf_result["passed"]

                # TF gate checks (12h→6h→1h)
                side = row_enriched.get("side", "long")
                gate = self.gates.evaluate(row_enriched, side)
                if not gate["passed"]:
                    continue

                # EVR
                side = row.get("side", "long")
                evr_result = self.evr_calc.compute_evr(row_enriched, side)
                evr_val = evr_result.get("evr", 0.0)
                median_r = evr_result.get("median_r", 0.0)
                row_enriched["evr"] = evr_val
                row_enriched["median_r"] = median_r
                row_enriched["stop_price"] = evr_result.get("stop_price", row_enriched.get("close"))

                # Tiering
                hazard_score = row_enriched.get("hazard", 0.0)
                regime_state = row_enriched.get("regime_state")
                tier_result = self.tiering.classify(
                    row=row_enriched,
                    confluence_pass=conf_pass,
                    evr_result={"evr": evr_val, "median_r": median_r},
                    hazard_score=hazard_score,
                    bar_index=bar_index,
                )

                cooling_gate = self.compound_cooling.allow_entry(
                    now=pd.Timestamp(t),
                    cooling_until=cooling_until,
                    conf=conf_score,
                    evr=evr_result,
                    hazard=danger["metrics"]["hazard"],
                )
                if not cooling_gate["allow"]:
                    continue

                if not tier_result["execute"]:
                    continue

                capital_out = self.capital.allocate(
                    equity=equity,
                    free_capital=free_capital,
                    locked_profit=locked_profit,
                    row=row_enriched.to_dict(),
                    mpc_manager=self.mpc,
                )
                if capital_out["lock_fraction"] > 0:
                    lock_delta = (equity - locked_profit) * capital_out["lock_fraction"]
                    locked_profit += max(lock_delta, 0.0)
                    free_capital = max(free_capital - max(lock_delta, 0.0), 0.0)

                pos_size_usd = capital_out["ticket_usd"]
                risk_perc = capital_out["risk_mode"]
                if pos_size_usd <= 0 or pos_size_usd > free_capital:
                    continue

                # Runner split
                runner_split = self.exec_cfg.get("runner_split", {})
                core_frac = runner_split.get("core_frac", 0.7)
                runner_frac = 1.0 - core_frac

                # Stop from EVR if available
                stop_price = evr_result.get("stop_price", row_enriched.get("stop_price", row_enriched["close"]))

                entry_price = row_enriched["close"]

                # Core leg
                core_usd = pos_size_usd * core_frac
                pos_core = self.simulator.open_position(asset, side, entry_price, core_usd, row_enriched["dt"], stop_price)
                pos_core.tier = tier_result["tier"]
                pos_core.conf = conf_score
                pos_core.evr = evr_val
                pos_core.risk = risk_perc
                pos_core.metadata["leg"] = "core"
                pos_core.metadata["median_r"] = median_r
                pos_core.metadata["initial_stop"] = stop_price
                pos_core.metadata["p_bos_cont"] = row_enriched.get("p_bos_cont", row_enriched.get("prob_bos_cont", 0.0))
                open_positions[pos_core.trade_id] = pos_core
                self.trade_log.append_open(
                    pos_core,
                    t,
                    tier_result["tier"],
                    conf_score,
                    evr_val,
                    risk_perc,
                    leg="core",
                    regime=regime_state,
                    session=row_enriched.get("session"),
                    hazard=hazard_score,
                    gates=gate.get("checks"),
                    gate_reasons=gate.get("reasons"),
                )

                # Runner leg
                if runner_frac > 0:
                    runner_usd = pos_size_usd * runner_frac
                    pos_runner = self.simulator.open_position(asset, side, entry_price, runner_usd, row_enriched["dt"], stop_price)
                    pos_runner.tier = tier_result["tier"]
                    pos_runner.conf = conf_score
                    pos_runner.evr = evr_val
                    pos_runner.risk = risk_perc
                    pos_runner.metadata["leg"] = "runner"
                    pos_runner.metadata["median_r"] = median_r
                    pos_runner.metadata["initial_stop"] = stop_price
                    pos_runner.metadata["p_bos_cont"] = row_enriched.get("p_bos_cont", row_enriched.get("prob_bos_cont", 0.0))
                    open_positions[pos_runner.trade_id] = pos_runner
                    self.trade_log.append_open(
                        pos_runner,
                        t,
                        tier_result["tier"],
                        conf_score,
                        evr_val,
                        risk_perc,
                        leg="runner",
                        regime=regime_state,
                        session=row_enriched.get("session"),
                        hazard=hazard_score,
                        gates=gate.get("checks"),
                        gate_reasons=gate.get("reasons"),
                    )

                free_capital -= pos_size_usd

                if self.dashboard:
                    self.dashboard.log_event("entry", pos_core.trade_id, {
                        "asset": asset,
                        "tier": pos_core.tier,
                        "confluence": conf_score,
                        "evr": evr_val,
                        "risk": risk_perc,
                        "leg": "core",
                    })
                    if runner_frac > 0:
                        self.dashboard.log_event("entry", pos_runner.trade_id, {
                            "asset": asset,
                            "tier": pos_runner.tier,
                            "confluence": conf_score,
                            "evr": evr_val,
                            "risk": risk_perc,
                            "leg": "runner",
                        })

            # Equity calc
            equity = self._portfolio_equity(asset_frames, open_positions, free_capital, locked_profit, t)

            max_equity = max(max_equity, equity)

            # Drawdown cooling
            dd = (max_equity - equity) / max_equity if max_equity else 0
            if (not self.compound_cooling.enabled) and dd >= self.exec_cfg.get("cooling_dd_trigger", 1.0):
                minutes = self.exec_cfg.get("cooling_minutes", 0)
                cooling_until = pd.Timestamp(t) + pd.Timedelta(minutes=minutes)

            # Dashboard equity update
            if self.dashboard:
                self.dashboard.update_state({
                    "timestamp": t,
                    "equity": equity,
                    "free_capital": free_capital,
                    "locked_profit": locked_profit,
                    "max_drawdown": dd,
                    "open_positions": len(open_positions),
                    "open_trades": open_positions,
                    "cooling_to": cooling_until.isoformat() if cooling_until is not None else None,
                })
            equity_history.append({
                "timestamp": pd.Timestamp(t),
                "equity": equity,
                "free_capital": free_capital,
                "locked_profit": locked_profit,
                "drawdown": dd,
                "open_positions": len(open_positions),
            })

        # Final metrics
        trade_df = self.trade_log.to_dataframe()
        equity_curve_df = pd.DataFrame(equity_history).drop_duplicates(subset=["timestamp"], keep="last")
        execution_log_df = build_timeline(trade_df, candles=self._primary_candles(asset_frames))
        primary_frame = self._primary_candles(asset_frames)
        metrics = BacktestMetrics(trade_df, starting_equity=float(self.exec_cfg.get("starting_equity", 0))).compute()
        if self.dashboard:
            self.dashboard.update_panels(metrics)
        result = {
            "trades": trade_df,
            "metrics": metrics,
            "equity_curve": equity_curve_df,
            "execution_log": execution_log_df,
            "candles": primary_frame[["timestamp", "dt", "open", "high", "low", "close", "volume"]].copy()
            if not primary_frame.empty and {"open", "high", "low", "close", "volume"}.issubset(primary_frame.columns)
            else primary_frame.copy(),
            "smc_features": primary_frame.copy(),
        }
        if self.dashboard and hasattr(self.dashboard, "update_backtest_bundle"):
            self.dashboard.update_backtest_bundle(result)

        LOG.info("[Backtester] Complete.")
        return result

    # ----------------------------------------------------------------------
    # MERGE MULTI-ASSET TIMELINES
    # ----------------------------------------------------------------------
    def _merge_timelines(self, asset_frames: Dict[str, pd.DataFrame]):
        ts = []
        for _, df in asset_frames.items():
            ts.extend(df["dt"].tolist())
        uniq = sorted(list(set(ts)))
        LOG.info(f"[Backtester] Timeline built with {len(uniq)} bars")
        return uniq

    # ----------------------------------------------------------------------
    def _portfolio_equity(
        self,
        asset_frames: Dict[str, pd.DataFrame],
        open_positions: Dict[str, Position],
        free_capital: float,
        locked_profit: float,
        ts,
    ) -> float:
        mtm = 0.0
        for pos in open_positions.values():
            px_series = asset_frames[pos.asset].loc[asset_frames[pos.asset]["dt"] == ts, "close"]
            px = float(px_series.values[0]) if not px_series.empty else float(pos.entry_price)
            mtm += self.simulator.mark_to_market(pos, px)
        return locked_profit + free_capital + mtm

    # ----------------------------------------------------------------------
    def _estimate_hazard(self, model, row: pd.Series) -> float:
        return float(row.get("hazard_score", row.get("hazard", 0.0)) or 0.0)

    def _primary_candles(self, asset_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not asset_frames:
            return pd.DataFrame()
        first_asset = next(iter(asset_frames))
        candles = asset_frames[first_asset].copy()
        if "timestamp" not in candles.columns and "dt" in candles.columns:
            candles["timestamp"] = pd.to_datetime(candles["dt"], errors="coerce")
        return candles

    # ----------------------------------------------------------------------
    def _inject_model_probs(self, row: pd.Series) -> pd.Series:
        """
        Attach model probabilities to the row for confluence/EVR usage.
        """
        enriched = row.copy()
        try:
            specialist_list = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
            preds = self.predictor.predict_single(row, specialist_list)
            for k, v in preds.items():
                if isinstance(v, dict):
                    # hazard_curve or other dict outputs
                    for kk, vv in v.items():
                        enriched[f"{k}_{kk}"] = vv
                else:
                    if k.startswith("prob_"):
                        key = f"p_{k.replace('prob_', '')}"
                        enriched[key] = v
                    else:
                        enriched[k] = v

            # Convenience aliases
            if "hazard_score" in preds:
                enriched["hazard"] = preds["hazard_score"]
        except Exception:
            pass
        return enriched

    # ----------------------------------------------------------------------
    def _compute_confluence(self, row: pd.Series) -> Dict[str, Any]:
        """
        Wrapper to support either evaluate() or compute_confluence() depending on engine definition.
        """
        if hasattr(self.confluence, "evaluate"):
            out = self.confluence.evaluate(row)
            score = out.get("confluence_score") or out.get("score") or 0.0
            passed = out.get("passed") if "passed" in out else out.get("allow", False)
            return {"score": score, "passed": bool(passed)}

        if hasattr(self.confluence, "compute_confluence"):
            try:
                out = self.confluence.compute_confluence({}, {}, None, self.cfg)
                score = out.get("score", 0.0)
                passed = out.get("allow", False)
                return {"score": score, "passed": passed}
            except Exception:
                return {"score": 0.0, "passed": True}

        return {"score": 0.0, "passed": True}
