"""
One-command forward-test room launcher (feature replay + live-style dashboard + telemetry).

Run:
    python run_BTCUSD_forward_live_room.py
"""

from __future__ import annotations

from python_bootstrap import ensure_runtime

ensure_runtime(("pandas", "streamlit", "fastapi", "uvicorn"))

import time
import os

from quant_system.cli.runners.forward_cli import main as forward_main
from quant_system.utils.logger import console_stage

from run_room_common import (
    ensure_file_exists,
    ensure_models_ready,
    log_elapsed,
    print_room_plan,
    run_runner,
    start_operator_ui,
    stop_processes,
)

ASSET = "BTCUSD"
FEATURES_CSV = "artifacts/features/BTCUSD/BTCUSD_features.csv"
OUT_DIR = "forward_outputs"
TERMINAL_HOST = "127.0.0.1"
TERMINAL_PORT = 8100
NEXT_PORT = 3000


def main() -> None:
    os.environ.setdefault("QUANT_RUNTIME_LOGS", "0")
    started_at = time.perf_counter()
    registry = ensure_models_ready()
    ensure_file_exists(
        FEATURES_CSV,
        "Build features first with: python build_BTCUSD_features.py",
    )
    print_room_plan(
        "BTCUSD Forward Live Room",
        asset=ASSET,
        terminal_host=TERMINAL_HOST,
        terminal_port=TERMINAL_PORT,
        out_dir=OUT_DIR,
        registry=registry,
        features_csv=FEATURES_CSV,
        next_port=NEXT_PORT,
    )

    ui_procs, ui_desc = start_operator_ui(
        terminal_host=TERMINAL_HOST,
        terminal_port=TERMINAL_PORT,
        next_port=NEXT_PORT,
    )
    keep_dashboard = True
    completed = False
    try:
        run_runner(
            forward_main,
            [
                "--asset",
                ASSET,
                "--features",
                FEATURES_CSV,
                "--out-dir",
                OUT_DIR,
                "--terminal-server",
                "--terminal-host",
                TERMINAL_HOST,
                "--terminal-port",
                str(TERMINAL_PORT),
            ],
        )
        console_stage("Forward run complete", f"artifacts={OUT_DIR}", status="ok")
        console_stage("Operator UI left running", f"close manually when done ({ui_desc})", status="info")
        keep_dashboard = False
        completed = True
    except KeyboardInterrupt:
        console_stage("Interrupted", "stopping forward room", status="warn")
    finally:
        log_elapsed(started_at, "Forward room runtime", ok=completed)
        if keep_dashboard:
            stop_processes(ui_procs)


if __name__ == "__main__":
    main()
