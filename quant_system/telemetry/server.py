from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from quant_system.dashboard.data_access import (
    load_model_registry_summary,
    resolve_mode_bundles,
    serialize_backtest_bundle,
    serialize_forward_bundle,
    serialize_model_summary,
)
from quant_system.telemetry.hub import get_telemetry_hub
from quant_system.telemetry.snapshot import REPO_ROOT, build_terminal_snapshot
from quant_system.utils.logger import get_logger

LOG = get_logger("telemetry_server")


def _resolve_dir(repo_root: Path, candidate: str | None, default_name: str) -> Path:
    if not candidate:
        return repo_root / default_name
    value = Path(candidate)
    return value if value.is_absolute() else (repo_root / value)


def create_app(repo_root: str | Path | None = None) -> FastAPI:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    hub = get_telemetry_hub()

    app = FastAPI(title="Quant SMC Terminal API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        hub.attach_loop(asyncio.get_running_loop())
        LOG.info("Telemetry server started repo_root=%s", root)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "transport": "fastapi + websocket",
            "repo_root": str(root),
            "has_live_snapshot": bool(hub.raw_snapshot()),
            "event_count": len(hub.recent_events()),
        }

    @app.get("/snapshot")
    async def snapshot() -> dict:
        return build_terminal_snapshot(hub.raw_snapshot(), repo_root=root)

    @app.get("/events")
    async def events(limit: int = Query(100, ge=1, le=5000)) -> dict:
        return {"events": hub.recent_events(limit)}

    @app.get("/dashboard/context")
    async def dashboard_context(
        backtest_dir: Optional[str] = Query(None),
        forward_dir: Optional[str] = Query(None),
        live_dir: Optional[str] = Query(None),
        model_dir: Optional[str] = Query(None),
        mode: str = Query("auto"),
    ) -> dict:
        raw_snapshot = hub.raw_snapshot()
        bt_root = _resolve_dir(root, backtest_dir, "backtest_outputs")
        fwd_root = _resolve_dir(root, forward_dir, "forward_outputs")
        live_root = _resolve_dir(root, live_dir, "live_outputs")
        mdl_root = _resolve_dir(root, model_dir, "models")

        bundles = resolve_mode_bundles(
            mode=mode,
            backtest_root=bt_root,
            forward_root=fwd_root,
            live_root=live_root,
            snapshot=raw_snapshot,
            adapter=None,
        )
        model_summary = load_model_registry_summary(mdl_root)
        model_version = model_summary["version"].max() if not model_summary.empty else "unavailable"
        return {
            "transport": "telemetry_api",
            "requested_mode": str(bundles["requested_mode"]),
            "resolved_mode": str(bundles["resolved_mode"]),
            "backtest_dir": str(bt_root),
            "forward_dir": str(fwd_root),
            "live_dir": str(live_root),
            "model_dir": str(mdl_root),
            "model_version": str(model_version),
            "backtest": serialize_backtest_bundle(bundles["backtest"]),
            "forward": serialize_forward_bundle(bundles["forward"]),
            "model_summary": serialize_model_summary(model_summary),
        }

    @app.websocket("/ws/terminal")
    async def terminal_ws(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        try:
            await websocket.send_json(
                {
                    "type": "terminal_snapshot",
                    "data": build_terminal_snapshot(hub.raw_snapshot(), repo_root=root),
                }
            )
            await hub.send_initial(websocket)
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(websocket)
        except Exception:
            await hub.disconnect(websocket)
            raise

    return app


app = create_app()
