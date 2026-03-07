import argparse

import uvicorn

from quant_system.telemetry.server import create_app
from quant_system.utils.logger import runtime_logged


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Quant SMC terminal FastAPI + WebSocket backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8100, type=int)
    parser.add_argument("--repo-root", default=None, help="Optional repo root override for artifact fallback.")
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


@runtime_logged("Terminal API runtime")
def main():
    args = parse_args()
    app = create_app(repo_root=args.repo_root)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
