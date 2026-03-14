import logging
import functools
import os
import time
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, TypeVar

try:  # pragma: no cover - optional pretty console
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None

_CONSOLE = Console() if Console is not None else None
F = TypeVar("F", bound=Callable)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Lightweight logger factory used across the codebase.
    Falls back to basicConfig once to avoid duplicate handlers.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            level=level,
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log(message: str, level: int = logging.INFO, logger: Optional[logging.Logger] = None):
    """
    Convenience wrapper for legacy log usage.
    """
    (logger or get_logger("quant_system")).log(level, message)


def fmt_ts(value) -> str:
    if value in (None, ""):
        return "-"
    try:
        ts = float(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def fmt_num(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except Exception:
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)


def fmt_seconds(value) -> str:
    if value is None:
        return "-"
    try:
        total = float(value)
    except Exception:
        return str(value)
    total = max(total, 0.0)
    if total < 60.0:
        return f"{total:.1f}s"
    rounded = int(round(total))
    days, rem = divmod(rounded, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def fmt_progress(completed, total, *, width: int = 24) -> str:
    try:
        done = max(int(completed), 0)
        whole = max(int(total), 1)
    except Exception:
        return "-"
    ratio = min(max(done / whole, 0.0), 1.0)
    width = max(int(width), 8)
    filled = int(round(ratio * width))
    if filled > width:
        filled = width
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}%"


def console_rule(title: str, *, style: str = "cyan"):
    if _CONSOLE is not None:
        _CONSOLE.rule(f"[bold {style}]{title}[/bold {style}]")
    else:
        print(f"==== {title} ====")


def console_stage(title: str, detail: Optional[str] = None, *, status: str = "info"):
    prefix = {
        "info": "INFO",
        "ok": "DONE",
        "warn": "WARN",
        "err": "ERR ",
    }.get(status, "INFO")
    text = f"{prefix} {title}"
    if detail:
        text = f"{text} | {detail}"
    if _CONSOLE is not None:
        color = {
            "info": "bright_cyan",
            "ok": "green",
            "warn": "yellow",
            "err": "red",
        }.get(status, "white")
        _CONSOLE.print(f"[bold {color}]{prefix}[/bold {color}] {title}" + (f" [dim]| {detail}[/dim]" if detail else ""))
    else:
        print(text)


def console_kv(title: str, items: Mapping[str, object], *, style: str = "cyan"):
    if _CONSOLE is not None and Table is not None:
        table = Table(title=title, title_style=f"bold {style}", show_header=False, box=None, pad_edge=False)
        table.add_column("k", style="bold white", no_wrap=True)
        table.add_column("v", style="white")
        for key, value in items.items():
            table.add_row(str(key), str(value))
        _CONSOLE.print(table)
    else:
        print(title)
        for key, value in items.items():
            print(f"  {key}: {value}")


def runtime_logged(label: str, *, ok_status: str = "ok", fail_status: str = "warn") -> Callable[[F], F]:
    """
    Decorator for executable entrypoints that prints standardized elapsed runtime.
    """

    def outer(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            started_at = time.perf_counter()
            completed = False
            try:
                result = fn(*args, **kwargs)
                completed = True
                return result
            finally:
                runtime_logs_enabled = os.getenv("QUANT_RUNTIME_LOGS", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if not runtime_logs_enabled:
                    return
                console_stage(
                    label,
                    f"elapsed={fmt_seconds(time.perf_counter() - started_at)}",
                    status=ok_status if completed else fail_status,
                )

        return inner  # type: ignore[return-value]

    return outer
