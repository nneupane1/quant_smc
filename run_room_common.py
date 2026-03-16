"""
Shared helpers for one-command runtime room launchers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import console_kv, console_rule, console_stage, fmt_seconds

REQUIRED_GENERIC_MODELS = [
    "liq_flow",
    "bos_cont",
    "flow_1h",
    "momo",
    "eop",
    "edp",
    "confluence_model",
    "hazard",
    "quantile",
]

REPO_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = REPO_ROOT / "frontend"


def resolve_registry_path(config_dir: str = "quant_system/config") -> Path:
    cfg = ConfigLoader(config_dir).load()
    paths_cfg = cfg.get("paths", {}) or {}
    models_cfg = cfg.get("models", {}) or {}
    return Path(
        paths_cfg.get("model_registry")
        or models_cfg.get("registry_path")
        or "models"
    )


def ensure_models_ready(config_dir: str = "quant_system/config") -> Path:
    registry = resolve_registry_path(config_dir)
    missing = [name for name in REQUIRED_GENERIC_MODELS if not (registry / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(
            f"Missing trained models in registry '{registry}': {missing_str}\n"
            "Train first (simple path):\n"
            "python -m quant_system.train_orchestrator --asset BTCUSD --tf-dir data/tf "
            "--features-out artifacts/features/BTCUSD/BTCUSD_features.csv "
            "--labels-out artifacts/labels/BTCUSD/BTCUSD_labels.csv "
            "--merged-out artifacts/train/BTCUSD/training_frame.csv "
            "--manifest-out artifacts/train/BTCUSD/train_manifest.json "
            "--model-state-out artifacts/train/BTCUSD/model_state.json"
        )
    return registry


def ensure_file_exists(path: str, hint: str) -> None:
    if not Path(path).exists():
        raise SystemExit(f"Missing required file: {path}\n{hint}")


def start_streamlit_dashboard(
    terminal_host: str = "127.0.0.1",
    terminal_port: int = 8100,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("QUANT_TERMINAL_API_BASE", f"http://{terminal_host}:{terminal_port}")
    cmd = [sys.executable, "-m", "streamlit", "run", "quant_system/dashboard/app.py"]
    return subprocess.Popen(cmd, env=env)


def _ensure_frontend_dependencies() -> None:
    if not FRONTEND_ROOT.exists():
        raise RuntimeError(f"Missing frontend directory: {FRONTEND_ROOT}")
    if (FRONTEND_ROOT / "node_modules").exists():
        return
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found; cannot start React terminal")
    console_stage("Frontend bootstrap", "installing npm dependencies (one-time)", status="info")
    proc = subprocess.run(
        [npm, "install"],
        cwd=str(FRONTEND_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("npm install failed in frontend/")


def start_next_terminal(
    *,
    terminal_host: str = "127.0.0.1",
    terminal_port: int = 8100,
    next_host: str = "127.0.0.1",
    next_port: int = 3000,
) -> subprocess.Popen:
    _ensure_frontend_dependencies()
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found; cannot start React terminal")

    env = os.environ.copy()
    env.setdefault("QUANT_TERMINAL_API_URL", f"http://{terminal_host}:{terminal_port}/snapshot")
    env.setdefault("NEXT_PUBLIC_TERMINAL_WS_URL", f"ws://{terminal_host}:{terminal_port}/ws/terminal")
    env.setdefault("QUANT_SMC_ROOT", str(REPO_ROOT))
    cmd = [npm, "run", "dev", "--", "--hostname", next_host, "--port", str(next_port)]
    return subprocess.Popen(cmd, cwd=str(FRONTEND_ROOT), env=env)


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def stop_processes(processes: Sequence[subprocess.Popen | None]) -> None:
    for proc in processes:
        stop_process(proc)


def run_runner(main_fn: Callable[[], None], argv: Sequence[str]) -> None:
    old_argv = list(sys.argv)
    sys.argv = [old_argv[0], *list(argv)]
    try:
        main_fn()
    finally:
        sys.argv = old_argv


def start_operator_ui(
    *,
    terminal_host: str = "127.0.0.1",
    terminal_port: int = 8100,
    next_host: str = "127.0.0.1",
    next_port: int = 3000,
) -> tuple[list[subprocess.Popen], str]:
    """
    Start operator UI surface.

    `QUANT_UI_SURFACE`:
      - `next` (default): React terminal, fallback to Streamlit on failure
      - `streamlit`: Streamlit only
      - `both`: React terminal + Streamlit side-by-side
    """
    mode = os.getenv("QUANT_UI_SURFACE", "next").strip().lower() or "next"
    processes: list[subprocess.Popen] = []
    endpoints: list[str] = []

    if mode not in {"next", "streamlit", "both"}:
        console_stage(
            "Unknown QUANT_UI_SURFACE",
            f"value='{mode}' is invalid; defaulting to 'next'",
            status="warn",
        )
        mode = "next"

    if mode in {"next", "both"}:
        try:
            next_proc = start_next_terminal(
                terminal_host=terminal_host,
                terminal_port=terminal_port,
                next_host=next_host,
                next_port=next_port,
            )
            time.sleep(2.0)
            if next_proc.poll() is not None:
                raise RuntimeError(f"React terminal exited early (code={next_proc.returncode})")
            processes.append(next_proc)
            endpoints.append(f"React terminal: http://{next_host}:{next_port}")
            console_stage("Operator UI", f"React terminal on http://{next_host}:{next_port}", status="ok")
        except Exception as exc:
            console_stage("React terminal failed", f"{exc}", status="warn")
            if mode == "next":
                streamlit_proc = start_streamlit_dashboard(terminal_host, terminal_port)
                processes.append(streamlit_proc)
                endpoints.append("Streamlit: http://localhost:8501")
                console_stage("Operator UI fallback", "Streamlit on http://localhost:8501", status="warn")

    if mode in {"streamlit", "both"}:
        streamlit_proc = start_streamlit_dashboard(terminal_host, terminal_port)
        processes.append(streamlit_proc)
        endpoints.append("Streamlit: http://localhost:8501")
        console_stage("Operator UI", "Streamlit on http://localhost:8501", status="ok")

    if not processes:
        raise RuntimeError("No operator UI surface started")

    return processes, "; ".join(endpoints)


def print_room_plan(
    title: str,
    *,
    asset: str,
    terminal_host: str,
    terminal_port: int,
    out_dir: str,
    registry: Path,
    features_csv: str | None = None,
    ui_mode: str | None = None,
    next_port: int = 3000,
) -> None:
    console_rule(title, style="bright_blue")
    mode = (ui_mode or os.getenv("QUANT_UI_SURFACE", "next")).strip().lower() or "next"
    if mode not in {"next", "streamlit", "both"}:
        mode = "next"

    if mode == "next":
        ui_endpoint = f"react http://127.0.0.1:{next_port} (streamlit fallback)"
    elif mode == "both":
        ui_endpoint = f"react http://127.0.0.1:{next_port} + streamlit http://localhost:8501"
    else:
        ui_endpoint = "streamlit http://localhost:8501"

    payload = {
        "asset": asset,
        "model_registry": str(registry),
        "terminal_api": f"http://{terminal_host}:{terminal_port}",
        "terminal_ws": f"ws://{terminal_host}:{terminal_port}/ws/terminal",
        "out_dir": out_dir,
        "operator_ui": ui_endpoint,
        "ui_mode": mode,
    }
    if features_csv is not None:
        payload["features_csv"] = features_csv
    console_kv("Run Card", payload, style="bright_blue")
    console_stage(
        "Telemetry endpoint",
        f"FastAPI snapshot on http://{terminal_host}:{terminal_port}/snapshot",
        status="info",
    )


def log_elapsed(started_at: float, title: str, *, ok: bool = True) -> None:
    runtime_logs_enabled = os.getenv("QUANT_RUNTIME_LOGS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not runtime_logs_enabled:
        return
    console_stage(
        title,
        f"elapsed={fmt_seconds(time.perf_counter() - started_at)}",
        status="ok" if ok else "warn",
    )
