import logging
from datetime import datetime, timezone
from typing import Mapping, Optional

try:  # pragma: no cover - optional pretty console
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None

_CONSOLE = Console() if Console is not None else None


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
        return f"{float(value):.1f}s"
    except Exception:
        return str(value)


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
