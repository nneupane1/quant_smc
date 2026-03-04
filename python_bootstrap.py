"""
Interpreter bootstrap for top-level launch scripts.

If the user invokes a launcher with the system Python that lacks repo
dependencies like pandas, re-exec into a better interpreter automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _candidate_interpreters() -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    env_candidates = [
        os.environ.get("CONDA_PREFIX") and str(Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"),
        os.environ.get("CONDA_PREFIX") and str(Path(os.environ["CONDA_PREFIX"]) / "bin" / "python3"),
        shutil.which("python"),
        shutil.which("python3"),
        "/Users/mac/opt/anaconda3/bin/python",
        "/Users/mac/opt/anaconda3/bin/python3",
    ]
    for cand in env_candidates:
        if not cand:
            continue
        path = str(Path(cand).expanduser())
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def _has_modules(python_exe: str, modules: Iterable[str]) -> bool:
    mod_list = ",".join(repr(m) for m in modules)
    code = (
        "import importlib.util,sys;"
        f"mods=[{mod_list}];"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "sys.exit(0 if not missing else 1)"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def ensure_runtime(modules: Iterable[str] = ("pandas",)) -> None:
    try:
        for mod in modules:
            __import__(mod)
        return
    except Exception:
        pass

    if os.environ.get("QUANT_SMC_BOOTSTRAPPED") == "1":
        missing = ", ".join(modules)
        raise SystemExit(
            f"Missing runtime dependencies ({missing}) in {sys.executable}. "
            "Install them in this interpreter or run the script with your conda/python environment."
        )

    current = str(Path(sys.executable).resolve())
    for candidate in _candidate_interpreters():
        if candidate == current:
            continue
        if _has_modules(candidate, modules):
            env = os.environ.copy()
            env["QUANT_SMC_BOOTSTRAPPED"] = "1"
            os.execve(candidate, [candidate, *sys.argv], env)

    missing = ", ".join(modules)
    raise SystemExit(
        f"Missing runtime dependencies ({missing}) in {sys.executable} and no compatible Python "
        "interpreter was found automatically."
    )
