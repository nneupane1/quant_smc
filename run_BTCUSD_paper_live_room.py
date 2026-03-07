"""
One-command paper-live room launcher (Kraken stream + live-style dashboard + telemetry).

Run:
    python run_BTCUSD_paper_live_room.py
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "streamlit", "fastapi", "uvicorn"))

import time
import os

from quant_system.cli.runners.live_cli import main as live_main
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import console_stage

from run_room_common import (
    ensure_models_ready,
    log_elapsed,
    print_room_plan,
    run_runner,
    start_operator_ui,
    stop_processes,
)

ASSET = "BTCUSD"
OUT_DIR = "live_outputs"
TERMINAL_HOST = "127.0.0.1"
TERMINAL_PORT = 8100
NEXT_PORT = 3000
CONFIG_DIR = "quant_system/config"


def _warn_if_live_orders_enabled() -> None:
    cfg = ConfigLoader(CONFIG_DIR).load()
    if bool(cfg.get("live_trading", {}).get("enabled", False)):
        console_stage(
            "Live trading enabled",
            "exchange order placement is ON in config (live_trading.enabled=true)",
            status="warn",
        )
    else:
        console_stage(
            "Paper mode",
            "live_trading.enabled=false, orders are simulated",
            status="ok",
        )


def main() -> None:
    os.environ.setdefault("QUANT_RUNTIME_LOGS", "0")
    started_at = time.perf_counter()
    registry = ensure_models_ready()
    print_room_plan(
        "BTCUSD Paper Live Room",
        asset=ASSET,
        terminal_host=TERMINAL_HOST,
        terminal_port=TERMINAL_PORT,
        out_dir=OUT_DIR,
        registry=registry,
        next_port=NEXT_PORT,
    )
    _warn_if_live_orders_enabled()

    ui_procs, ui_desc = start_operator_ui(
        terminal_host=TERMINAL_HOST,
        terminal_port=TERMINAL_PORT,
        next_port=NEXT_PORT,
    )
    console_stage("Operator UI active", ui_desc, status="info")
    completed = False
    try:
        run_runner(
            live_main,
            [
                "--asset",
                ASSET,
                "--stream",
                "--out-dir",
                OUT_DIR,
                "--terminal-server",
                "--terminal-host",
                TERMINAL_HOST,
                "--terminal-port",
                str(TERMINAL_PORT),
            ],
        )
        completed = True
    except KeyboardInterrupt:
        console_stage("Interrupted", "stopping live room", status="warn")
    finally:
        log_elapsed(started_at, "Paper live room runtime", ok=completed)
        stop_processes(ui_procs)


if __name__ == "__main__":
    main()
