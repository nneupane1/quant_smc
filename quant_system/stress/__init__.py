"""
Deterministic stress-testing utilities.
"""

from typing import Any

__all__ = ["run_stress_matrix"]


def run_stress_matrix(*args: Any, **kwargs: Any):
    from .deterministic_matrix import run_stress_matrix as _run_stress_matrix

    return _run_stress_matrix(*args, **kwargs)
