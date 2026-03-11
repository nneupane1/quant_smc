"""
One-command deterministic stress matrix launcher.

Run:
    python run_BTCUSD_stress_matrix.py
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "numpy"))

import time

from quant_system.stress.deterministic_matrix import StressThresholds, run_stress_matrix
from quant_system.utils.logger import console_stage, fmt_seconds


def main() -> None:
    started_at = time.perf_counter()
    completed = False
    try:
        run_stress_matrix(
            backtest_dir="backtest_outputs",
            out_dir="backtest_outputs/stress_matrix",
            thresholds=StressThresholds(
                max_drawdown_pct=0.40,
                max_abs_cvar95=0.03,
                min_ending_equity_pct=0.75,
                max_ror_proxy=0.80,
            ),
            enforce=False,
        )
        completed = True
    except FileNotFoundError as exc:
        console_stage("Stress launcher", str(exc), status="err")
        raise SystemExit(1) from exc
    finally:
        console_stage(
            "Stress launcher runtime",
            f"elapsed={fmt_seconds(time.perf_counter() - started_at)}",
            status="ok" if completed else "warn",
        )


if __name__ == "__main__":
    main()
