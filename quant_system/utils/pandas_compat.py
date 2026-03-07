"""Pandas compatibility helpers for third-party libraries."""

from __future__ import annotations


def ensure_stringmethods_alias() -> None:
    """
    LightGBM expects `pandas.core.strings.StringMethods` in some environments.
    Newer pandas exposes it under `pandas.core.strings.accessor.StringMethods`.
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
        # Best effort only.
        return
