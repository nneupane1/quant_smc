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


def _patch_pandas_for_lightgbm() -> None:
    """
    Best-effort pandas compatibility shim for LightGBM import.
    Some pandas versions expose StringMethods under accessor only.
    """
    try:
        import pandas as pd
    except Exception:
        return
    try:
        strings_mod = getattr(getattr(pd, "core", None), "strings", None)
        if strings_mod is None:
            return
        if hasattr(strings_mod, "StringMethods"):
            return
        accessor = getattr(strings_mod, "accessor", None)
        cls = getattr(accessor, "StringMethods", None) if accessor is not None else None
        if cls is not None:
            setattr(strings_mod, "StringMethods", cls)
    except Exception:
        return


def _can_import_module(mod: str) -> bool:
    try:
        if mod == "lightgbm":
            _patch_pandas_for_lightgbm()
        __import__(mod)
        return True
    except Exception:
        return False


def _missing_modules(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for mod in modules:
        if not _can_import_module(mod):
            missing.append(mod)
    return missing


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
        "import sys;"
        "def _patch():\n"
        "    try:\n"
        "        import pandas as pd\n"
        "        strings_mod = getattr(getattr(pd, 'core', None), 'strings', None)\n"
        "        if strings_mod is not None and not hasattr(strings_mod, 'StringMethods'):\n"
        "            accessor = getattr(strings_mod, 'accessor', None)\n"
        "            cls = getattr(accessor, 'StringMethods', None) if accessor is not None else None\n"
        "            if cls is not None:\n"
        "                setattr(strings_mod, 'StringMethods', cls)\n"
        "    except Exception:\n"
        "        pass\n"
        f"mods=[{mod_list}];"
        "missing=[];"
        "for m in mods:\n"
        "    try:\n"
        "        if m == 'lightgbm':\n"
        "            _patch()\n"
        "        __import__(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
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
    required = tuple(dict.fromkeys(modules))
    missing = _missing_modules(required)
    if not missing:
        return

    if os.environ.get("QUANT_SMC_BOOTSTRAPPED") == "1":
        missing_msg = ", ".join(missing)
        raise SystemExit(
            f"Runtime dependency import failed ({missing_msg}) in {sys.executable}. "
            "Install them in this interpreter or run the script with your conda/python environment."
        )

    current = str(Path(sys.executable).resolve())
    for candidate in _candidate_interpreters():
        if candidate == current:
            continue
        if _has_modules(candidate, required):
            env = os.environ.copy()
            env["QUANT_SMC_BOOTSTRAPPED"] = "1"
            os.execve(candidate, [candidate, *sys.argv], env)

    missing_msg = ", ".join(missing)
    raise SystemExit(
        f"Runtime dependency import failed ({missing_msg}) in {sys.executable} and no compatible Python "
        "interpreter was found automatically."
    )
