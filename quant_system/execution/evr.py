"""Compatibility wrappers for the authoritative EVR implementation."""

from __future__ import annotations

from typing import Any, Dict, Union

from quant_system.execution.gating.evr import EVRCalculator as _EVRCalculator


class EVRCalculator(_EVRCalculator):
    """Backwards-compatible alias to the canonical gating EVR calculator."""


class EVREngine:
    """
    Legacy compatibility shim.

    Older code referenced `quant_system.execution.evr.EVREngine`. The canonical
    implementation now lives in `quant_system.execution.gating.evr.EVRCalculator`.
    """

    def __init__(self, config: Union[Dict[str, Any], Any]):
        self._calc = EVRCalculator(config)

    def compute_evr(self, *args, **kwargs):
        return self._calc.compute(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self._calc.compute(*args, **kwargs)
