"""
performance_memory.py
Stores historical performance statistics for each model version:
 • EVR
 • precision
 • max_dd
 • cvar95

Ranked selection used by ModelSelector fallback logic.
"""

from pathlib import Path
import json
import time
from quant_system.utils.logger import get_logger

LOG = get_logger("performance_memory")


class PerformanceMemory:

    def __init__(self, path: str = "model_performance.json", defaults: dict = None):
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps(defaults or {}, indent=2))
        self.data = self._load()

    def _load(self):
        try:
            payload = json.loads(self.path.read_text() or "{}")
            return payload if isinstance(payload, dict) else {}
        except Exception:
            LOG.warning("[PerfMem] Failed to read %s; starting empty.", self.path)
            return {}

    # --------------------------------------------------------------
    def update(self, model_id: str, stats: dict):
        LOG.info(f"[PerfMem] update {model_id}: {stats}")
        prev = self.data.get(model_id, {})
        merged = {**prev, **(stats or {})}
        merged["updated_at"] = time.time()
        self.data[model_id] = merged
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, model_id: str):
        return self.data.get(model_id)

    # --------------------------------------------------------------
    def rank_models(self, available_models=None):
        """
        Rank by:
          1) higher EVR
          2) lower CVaR
          3) higher precision
        """
        if not self.data:
            LOG.warning("[PerfMem] No performance data; returning empty rank list.")
            return []

        available = set(available_models or [])

        ranked = []
        for m, stats in self.data.items():
            if available and m not in available:
                continue
            ranked.append({
                "model_id": m,
                "evr": stats.get("evr", 0.0),
                "precision": stats.get("precision", 0.0),
                "dd": stats.get("max_dd", 999.0),
                "cvar": stats.get("cvar95", 999.0),
                "updated_at": stats.get("updated_at"),
            })

        ranked = sorted(
            ranked,
            key=lambda x: (-x["evr"], x["cvar"], -x["precision"])
        )

        LOG.info(f"[PerfMem] ranking: {ranked}")
        return ranked
