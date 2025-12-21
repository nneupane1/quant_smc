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
from quant_system.utils.logger import get_logger

LOG = get_logger("performance_memory")


class PerformanceMemory:

    def __init__(self, path: str = "model_performance.json", defaults: dict = None):
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps(defaults or {}))
        self.data = json.loads(self.path.read_text() or "{}")

    # --------------------------------------------------------------
    def update(self, model_id: str, stats: dict):
        LOG.info(f"[PerfMem] update {model_id}: {stats}")
        self.data[model_id] = stats
        self.path.write_text(json.dumps(self.data, indent=2))

    # --------------------------------------------------------------
    def rank_models(self):
        """
        Rank by:
          1) higher EVR
          2) lower CVaR
          3) higher precision
        """
        if not self.data:
            LOG.warning("[PerfMem] No performance data; returning empty rank list.")
            return []

        ranked = []
        for m, stats in self.data.items():
            ranked.append({
                "model_id": m,
                "evr": stats.get("evr", 0.0),
                "precision": stats.get("precision", 0.0),
                "dd": stats.get("max_dd", 999.0),
                "cvar": stats.get("cvar95", 999.0),
            })

        ranked = sorted(
            ranked,
            key=lambda x: (-x["evr"], x["cvar"], -x["precision"])
        )

        LOG.info(f"[PerfMem] ranking: {ranked}")
        return ranked
