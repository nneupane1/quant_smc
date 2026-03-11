"""
Deterministic stress matrix runner.

This module replays backtest trade ledger PnL through fixed shock scenarios
to validate robustness without adding randomness (no Monte Carlo).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_num, fmt_seconds


@dataclass(frozen=True)
class StressScenario:
    name: str
    description: str
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    fee_bps: float = 0.0
    fill_ratio: float = 1.0
    latency_bars: int = 0
    latency_penalty_r: float = 0.0
    winner_haircut: float = 1.0
    loss_multiplier: float = 1.0
    pnl_shift_usd: float = 0.0


@dataclass(frozen=True)
class StressThresholds:
    # Default values are intentionally moderate so this remains a risk audit,
    # not a hyper-conservative blocker for alert generation.
    max_drawdown_pct: float = 0.40
    max_abs_cvar95: float = 0.03
    min_ending_equity_pct: float = 0.75
    max_ror_proxy: float = 0.80


def default_scenarios() -> List[StressScenario]:
    return [
        StressScenario(name="baseline", description="Unchanged backtest ledger."),
        StressScenario(
            name="cost_pressure",
            description="Higher friction: slippage+spread+fees.",
            slippage_bps=3.0,
            spread_bps=4.0,
            fee_bps=2.0,
        ),
        StressScenario(
            name="latency_1bar",
            description="Execution delay impact on realized R.",
            latency_bars=1,
            latency_penalty_r=0.15,
        ),
        StressScenario(
            name="partial_fill",
            description="Partial fills plus moderate friction and delay.",
            fill_ratio=0.75,
            slippage_bps=5.0,
            spread_bps=6.0,
            fee_bps=2.0,
            latency_bars=1,
            latency_penalty_r=0.10,
        ),
        StressScenario(
            name="adverse_tail",
            description="Winners compress, losers extend.",
            slippage_bps=5.0,
            spread_bps=6.0,
            fee_bps=2.0,
            winner_haircut=0.90,
            loss_multiplier=1.25,
        ),
        StressScenario(
            name="hard_regime",
            description="Compound adverse regime shock (deterministic worst case).",
            fill_ratio=0.65,
            slippage_bps=8.0,
            spread_bps=10.0,
            fee_bps=3.0,
            latency_bars=2,
            latency_penalty_r=0.20,
            winner_haircut=0.85,
            loss_multiplier=1.35,
        ),
    ]


def _coerce_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _load_ledger(backtest_dir: Path) -> pd.DataFrame:
    ledger_path = backtest_dir / "ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"Missing ledger at {ledger_path}. Run a backtest first, e.g. "
            "python run_BTCUSD_backtest_live_room.py"
        )
    df = pd.read_csv(ledger_path, parse_dates=["entry_ts", "exit_ts"])
    if df.empty:
        raise ValueError(f"Ledger is empty: {ledger_path}")
    if "pnl" not in df.columns:
        raise ValueError(f"Ledger missing required column 'pnl': {ledger_path}")

    for col in ("size_usd", "qty", "entry_price", "stop_price", "pnl"):
        if col not in df.columns:
            df[col] = 0.0

    if "entry_ts" not in df.columns:
        df["entry_ts"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    df = df.sort_values("entry_ts").reset_index(drop=True)
    return df


def _load_starting_equity(backtest_dir: Path, cli_equity: float | None) -> float:
    if cli_equity is not None and cli_equity > 0:
        return float(cli_equity)

    summary_path = backtest_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
            val = float(payload.get("starting_equity", 0.0) or 0.0)
            if val > 0:
                return val
        except Exception:
            pass

    return 20_000.0


def _apply_scenario(ledger: pd.DataFrame, scenario: StressScenario) -> pd.DataFrame:
    out = ledger.copy()
    base_pnl = _coerce_num(out["pnl"], 0.0).to_numpy(dtype=float)

    size_usd = np.abs(_coerce_num(out["size_usd"], 0.0).to_numpy(dtype=float))
    qty = np.abs(_coerce_num(out["qty"], 0.0).to_numpy(dtype=float))
    entry = _coerce_num(out["entry_price"], 0.0).to_numpy(dtype=float)
    stop = _coerce_num(out["stop_price"], 0.0).to_numpy(dtype=float)

    one_r = np.abs(entry - stop) * qty
    fallback_one_r = np.maximum(size_usd * 0.01, 1e-9)
    one_r = np.where(one_r > 1e-9, one_r, fallback_one_r)

    fill_ratio = float(np.clip(scenario.fill_ratio, 0.0, 1.0))
    total_cost_bps = float(max(scenario.slippage_bps + scenario.spread_bps + scenario.fee_bps, 0.0))
    friction_cost = size_usd * (total_cost_bps / 10_000.0) * fill_ratio

    adj_pnl = base_pnl * fill_ratio - friction_cost

    if scenario.latency_bars > 0 and scenario.latency_penalty_r > 0:
        latency_penalty = one_r * float(scenario.latency_bars) * float(scenario.latency_penalty_r)
        adj_pnl = np.where(adj_pnl >= 0.0, adj_pnl - latency_penalty, adj_pnl - (latency_penalty * 0.5))

    if scenario.winner_haircut != 1.0:
        adj_pnl = np.where(adj_pnl > 0.0, adj_pnl * float(scenario.winner_haircut), adj_pnl)
    if scenario.loss_multiplier != 1.0:
        adj_pnl = np.where(adj_pnl < 0.0, adj_pnl * float(scenario.loss_multiplier), adj_pnl)
    if scenario.pnl_shift_usd != 0.0:
        adj_pnl = adj_pnl + float(scenario.pnl_shift_usd)

    out["scenario"] = scenario.name
    out["adj_pnl"] = adj_pnl
    return out


def _risk_of_ruin_proxy(pnls: np.ndarray, starting_equity: float) -> float:
    if pnls.size == 0:
        return 1.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if wins.size == 0 or losses.size == 0:
        return 1.0 if losses.size > 0 else 0.0

    win_rate = wins.size / pnls.size
    avg_win = float(wins.mean())
    avg_loss = float(abs(losses.mean()))
    if avg_win <= 0 or avg_loss <= 0:
        return 1.0

    payoff = avg_win / avg_loss
    edge = win_rate - (1.0 - win_rate) / max(payoff, 1e-9)
    if edge <= 0:
        return 1.0

    units = float(np.clip(starting_equity / max(avg_loss, 1e-9), 1.0, 200.0))
    # Deterministic approximation: larger positive edge and larger capital units
    # lower ruin probability exponentially.
    ror = math.exp(-2.0 * edge * units)
    return float(np.clip(ror, 0.0, 1.0))


def _compute_metrics(adjusted_ledger: pd.DataFrame, starting_equity: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    pnls = _coerce_num(adjusted_ledger["adj_pnl"], 0.0).to_numpy(dtype=float)
    n = int(pnls.size)
    if n == 0:
        curve = pd.DataFrame({"step": [0], "equity": [starting_equity], "drawdown_pct": [0.0]})
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "ending_equity": float(starting_equity),
            "max_drawdown_pct": 0.0,
            "var95": 0.0,
            "cvar95": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "risk_of_ruin_proxy": 1.0,
        }, curve

    equity = np.empty(n + 1, dtype=float)
    equity[0] = float(starting_equity)
    returns = np.empty(n, dtype=float)
    peaks = np.empty(n + 1, dtype=float)
    peaks[0] = float(starting_equity)

    for i, pnl in enumerate(pnls, start=1):
        prev_equity = max(equity[i - 1], 1e-9)
        returns[i - 1] = pnl / prev_equity
        equity[i] = equity[i - 1] + pnl
        peaks[i] = max(peaks[i - 1], equity[i])

    drawdown_pct = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd = float(np.max(drawdown_pct))
    var95 = float(np.quantile(returns, 0.05))
    tail = returns[returns <= var95]
    cvar95 = float(tail.mean()) if tail.size else var95

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    payoff = float(avg_win / abs(avg_loss)) if avg_win > 0 and avg_loss < 0 else 0.0
    ror_proxy = _risk_of_ruin_proxy(pnls, float(starting_equity))

    metrics = {
        "trades": n,
        "win_rate": float((pnls > 0).mean()),
        "total_pnl": float(pnls.sum()),
        "ending_equity": float(equity[-1]),
        "max_drawdown_pct": max_dd,
        "var95": var95,
        "cvar95": cvar95,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "risk_of_ruin_proxy": ror_proxy,
    }
    curve = pd.DataFrame(
        {
            "step": np.arange(0, n + 1),
            "equity": equity,
            "drawdown_pct": drawdown_pct,
        }
    )
    return metrics, curve


def _evaluate_thresholds(
    metrics: Dict[str, float],
    thresholds: StressThresholds,
    starting_equity: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if metrics["max_drawdown_pct"] > thresholds.max_drawdown_pct:
        reasons.append("max_drawdown")
    if abs(metrics["cvar95"]) > thresholds.max_abs_cvar95:
        reasons.append("cvar95")
    if metrics["ending_equity"] < (starting_equity * thresholds.min_ending_equity_pct):
        reasons.append("ending_equity_floor")
    if metrics["risk_of_ruin_proxy"] > thresholds.max_ror_proxy:
        reasons.append("risk_of_ruin_proxy")
    return len(reasons) == 0, reasons


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def run_stress_matrix(
    *,
    backtest_dir: str = "backtest_outputs",
    out_dir: str | None = None,
    starting_equity: float | None = None,
    thresholds: StressThresholds | None = None,
    enforce: bool = False,
) -> Dict[str, object]:
    started_at = time.perf_counter()
    thresholds = thresholds or StressThresholds()
    bt_dir = Path(backtest_dir)
    if not bt_dir.exists():
        # Fallback used by older CLI path.
        fallback = Path("artifacts/backtest/latest")
        if fallback.exists():
            bt_dir = fallback
        else:
            raise FileNotFoundError(
                f"Backtest directory not found: {backtest_dir}. "
                "Run `python run_BTCUSD_backtest_live_room.py` first."
            )

    target_dir = Path(out_dir) if out_dir else bt_dir / "stress_matrix"
    target_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = target_dir / "equity_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    ledger = _load_ledger(bt_dir)
    equity0 = _load_starting_equity(bt_dir, starting_equity)
    scenarios = default_scenarios()

    console_rule("Deterministic Stress Matrix", style="bright_magenta")
    console_kv(
        "Stress Plan",
        {
            "backtest_dir": str(bt_dir),
            "trades": fmt_num(len(ledger)),
            "starting_equity": fmt_num(equity0),
            "scenarios": ", ".join(s.name for s in scenarios),
            "enforce": bool(enforce),
            "max_dd": _format_pct(thresholds.max_drawdown_pct),
            "max_abs_cvar95": _format_pct(thresholds.max_abs_cvar95),
            "min_end_equity": _format_pct(thresholds.min_ending_equity_pct),
            "max_ror_proxy": _format_pct(thresholds.max_ror_proxy),
        },
        style="bright_magenta",
    )

    rows: List[Dict[str, object]] = []
    baseline_pnl: float | None = None
    baseline_eq: float | None = None

    for idx, scenario in enumerate(scenarios, start=1):
        console_stage(
            f"Scenario {idx}/{len(scenarios)}",
            f"{scenario.name} | {scenario.description}",
            status="info",
        )
        stressed = _apply_scenario(ledger, scenario)
        metrics, curve = _compute_metrics(stressed, equity0)
        passed, reasons = _evaluate_thresholds(metrics, thresholds, equity0)

        if scenario.name == "baseline":
            baseline_pnl = metrics["total_pnl"]
            baseline_eq = metrics["ending_equity"]

        curve.to_csv(curves_dir / f"{scenario.name}.csv", index=False)
        row = {
            "scenario": scenario.name,
            "description": scenario.description,
            "trades": int(metrics["trades"]),
            "win_rate": float(metrics["win_rate"]),
            "total_pnl": float(metrics["total_pnl"]),
            "ending_equity": float(metrics["ending_equity"]),
            "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
            "var95": float(metrics["var95"]),
            "cvar95": float(metrics["cvar95"]),
            "payoff_ratio": float(metrics["payoff_ratio"]),
            "risk_of_ruin_proxy": float(metrics["risk_of_ruin_proxy"]),
            "passed": bool(passed),
            "fail_reasons": ",".join(reasons),
        }
        rows.append(row)
        console_stage(
            f"Scenario {scenario.name} {'PASS' if passed else 'FAIL'}",
            (
                f"end_eq={fmt_num(row['ending_equity'])} "
                f"max_dd={_format_pct(row['max_drawdown_pct'])} "
                f"cvar95={_format_pct(abs(row['cvar95']))}"
            ),
            status="ok" if passed else "warn",
        )

    report_df = pd.DataFrame(rows)
    if baseline_pnl is not None:
        report_df["delta_pnl_vs_baseline"] = report_df["total_pnl"] - float(baseline_pnl)
    if baseline_eq is not None:
        report_df["delta_eq_vs_baseline"] = report_df["ending_equity"] - float(baseline_eq)

    report_df = report_df.sort_values(["passed", "ending_equity"], ascending=[False, False]).reset_index(drop=True)
    report_df.to_csv(target_dir / "stress_summary.csv", index=False)

    gate = {
        "all_passed": bool(report_df["passed"].all()) if not report_df.empty else False,
        "failed_scenarios": report_df.loc[~report_df["passed"], "scenario"].tolist() if not report_df.empty else [],
    }
    payload = {
        "backtest_dir": str(bt_dir),
        "out_dir": str(target_dir),
        "starting_equity": equity0,
        "thresholds": asdict(thresholds),
        "scenarios": [asdict(s) for s in scenarios],
        "gate": gate,
        "runtime_seconds": float(time.perf_counter() - started_at),
    }
    (target_dir / "stress_manifest.json").write_text(json.dumps(payload, indent=2))

    console_stage(
        "Stress matrix complete",
        f"all_passed={gate['all_passed']} failed={len(gate['failed_scenarios'])} out={target_dir}",
        status="ok" if gate["all_passed"] else "warn",
    )
    console_stage(
        "Stress runtime",
        f"elapsed={fmt_seconds(payload['runtime_seconds'])}",
        status="info",
    )

    if enforce and not gate["all_passed"]:
        raise SystemExit(
            "Deterministic stress gate failed for scenarios: "
            + ", ".join(gate["failed_scenarios"])
        )

    return {
        "summary_path": str(target_dir / "stress_summary.csv"),
        "manifest_path": str(target_dir / "stress_manifest.json"),
        "gate": gate,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run deterministic stress scenarios on backtest trade ledger.")
    p.add_argument("--backtest-dir", default="backtest_outputs", help="Directory containing backtest artifacts (ledger.csv).")
    p.add_argument("--out-dir", default=None, help="Optional output directory. Defaults to <backtest-dir>/stress_matrix.")
    p.add_argument("--starting-equity", type=float, default=None, help="Override starting equity (defaults to summary.json or 20000).")
    p.add_argument("--max-dd", type=float, default=0.40, help="Max allowed drawdown pct (decimal). Default 0.40.")
    p.add_argument("--max-cvar95", type=float, default=0.03, help="Max allowed |CVaR95| per-trade return. Default 0.03.")
    p.add_argument("--min-ending-equity-pct", type=float, default=0.75, help="Minimum ending equity ratio vs start. Default 0.75.")
    p.add_argument("--max-ror-proxy", type=float, default=0.80, help="Max risk-of-ruin proxy in [0,1]. Default 0.80.")
    p.add_argument("--enforce", action="store_true", help="Exit non-zero if any scenario fails.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = StressThresholds(
        max_drawdown_pct=float(args.max_dd),
        max_abs_cvar95=float(args.max_cvar95),
        min_ending_equity_pct=float(args.min_ending_equity_pct),
        max_ror_proxy=float(args.max_ror_proxy),
    )
    run_stress_matrix(
        backtest_dir=args.backtest_dir,
        out_dir=args.out_dir,
        starting_equity=args.starting_equity,
        thresholds=thresholds,
        enforce=bool(args.enforce),
    )


if __name__ == "__main__":
    main()
