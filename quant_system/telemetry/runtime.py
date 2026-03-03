from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import uvicorn

from quant_system.telemetry.server import create_app
from quant_system.utils.logger import get_logger

LOG = get_logger("telemetry_runtime")


@dataclass
class TerminalServerHandle:
    host: str
    port: int
    server: uvicorn.Server
    thread: threading.Thread

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws/terminal"


def start_terminal_server(host: str = "127.0.0.1", port: int = 8100, repo_root: Optional[str] = None) -> TerminalServerHandle:
    app = create_app(repo_root=repo_root)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="quant-terminal-api", daemon=True)
    thread.start()
    LOG.info("Terminal server thread started host=%s port=%s", host, port)
    return TerminalServerHandle(host=host, port=port, server=server, thread=thread)
