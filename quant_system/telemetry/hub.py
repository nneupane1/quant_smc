from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

import pandas as pd

from quant_system.utils.logger import get_logger

LOG = get_logger("telemetry_hub")


class TelemetryHub:
    def __init__(self, max_events: int = 2000):
        self.max_events = max_events
        self._raw_snapshot: Dict[str, Any] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._clients: Set[Any] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def raw_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._raw_snapshot)

    def recent_events(self, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        if limit is None:
            return events
        return events[-max(int(limit), 0):]

    def publish_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._raw_snapshot = dict(snapshot or {})
        from quant_system.telemetry.snapshot import build_terminal_snapshot

        self._schedule_broadcast(
            {
                "type": "terminal_snapshot",
                "data": build_terminal_snapshot(self.raw_snapshot()),
            }
        )

    def publish_event(self, event: Dict[str, Any]) -> None:
        event = self._jsonable(event)
        with self._lock:
            self._events.append(dict(event or {}))
            if self._raw_snapshot:
                events = list(self._raw_snapshot.get("events", []))
                events.append(dict(event or {}))
                self._raw_snapshot["events"] = events[-self.max_events :]
        self._schedule_broadcast({"type": "terminal_event", "data": dict(event or {})})

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        with self._lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket: Any) -> None:
        with self._lock:
            self._clients.discard(websocket)

    async def send_initial(self, websocket: Any) -> None:
        from quant_system.telemetry.snapshot import build_terminal_snapshot

        await websocket.send_json(
            {
                "type": "bootstrap",
                "data": {
                    "snapshot": build_terminal_snapshot(self.raw_snapshot()),
                    "events": self._jsonable(self.recent_events()),
                },
            }
        )

    def _schedule_broadcast(self, payload: Dict[str, Any]) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Telemetry broadcast scheduling failed: %s", exc)

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        dead = []
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        if dead:
            with self._lock:
                for websocket in dead:
                    self._clients.discard(websocket)

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.DataFrame):
            return [cls._jsonable(rec) for rec in value.to_dict(orient="records")]
        if isinstance(value, pd.Series):
            return {str(k): cls._jsonable(v) for k, v in value.to_dict().items()}
        if isinstance(value, dict):
            return {str(k): cls._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, deque)):
            return [cls._jsonable(v) for v in value]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "__dict__"):
            return {str(k): cls._jsonable(v) for k, v in vars(value).items()}
        try:
            return float(value)
        except Exception:
            return str(value)


_GLOBAL_HUB: Optional[TelemetryHub] = None


def get_telemetry_hub() -> TelemetryHub:
    global _GLOBAL_HUB
    if _GLOBAL_HUB is None:
        _GLOBAL_HUB = TelemetryHub()
    return _GLOBAL_HUB
