"""
ModelGovernor:
 - Evaluates new model submissions against metric/calibration/risk thresholds
 - Logs submissions, shadow runs, promotions, rollbacks
 - Hooks into ModelRegistry to flip active models
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List

from quant_system.utils.logger import get_logger

LOG = get_logger("model_governor")


class ModelGovernor:
    def __init__(self, registry, config: Dict[str, Any]):
        self.registry = registry
        self.config = config or {}
        self.log_path = Path(self.config.get("governance_log", "governance_log.json"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.slot = self.config.get("deployment_slot", "production")
        self.defaults = {
            "min_requirements": {
                "pr_auc": 0.0,
                "brier": 1.0,
                "ece": 1.0,
                "max_dd": 1e9,
                "cvar95": 1e9,
            },
            "shadow_requirements": {
                "min_evr_delta": -999.0,
                "max_dd_delta": 999.0,
                "min_precision": 0.0,
            },
        }

    # ------------------------------------------------------------------
    def submit(self, model_id: str, metrics: dict, risk: dict, calib: dict):
        """
        Submit a newly trained model version.
        metrics: {pr_auc, brier, acc, f1, ...}
        risk:    {max_dd, cvar95, hazard_ib_score}
        calib:   {ece, reliability_slope, ...}
        """
        LOG.info(f"[Gov] submission received: {model_id}")

        payload = {
            "timestamp": time.time(),
            "model_id": model_id,
            "metrics": metrics,
            "risk": risk,
            "calibration": calib,
        }
        self._append_log("submission", payload)

        return self._evaluate(model_id, metrics, risk, calib)

    # ------------------------------------------------------------------
    def _evaluate(self, model_id, metrics, risk, calib):
        """
        Evaluate submission using config thresholds.
        """
        cfg = {**self.defaults["min_requirements"], **self.config.get("min_requirements", {})}
        pass_metrics = (
            metrics.get("pr_auc", 0.0) >= cfg.get("pr_auc", 0.0)
            and metrics.get("brier", 1.0) <= cfg.get("brier", 1.0)
        )

        pass_calib = calib.get("ece", 1.0) <= cfg.get("ece", 1.0)

        pass_risk = (
            risk.get("max_dd", 1e9) <= cfg.get("max_dd", 1e9)
            and risk.get("cvar95", 1e9) <= cfg.get("cvar95", 1e9)
        )

        ok = pass_metrics and pass_calib and pass_risk

        LOG.info(
            f"[Gov] evaluation {model_id}: "
            f"metrics={pass_metrics}, calib={pass_calib}, risk={pass_risk}"
        )

        decision = {
            "approved": bool(ok),
            "checks": {
                "metrics": pass_metrics,
                "calibration": pass_calib,
                "risk": pass_risk,
            },
        }
        self._append_log("evaluation", {"timestamp": time.time(), "model_id": model_id, **decision})
        return decision

    # ------------------------------------------------------------------
    def approve_shadow(self, model_id):
        """Mark a model version as approved for shadow-mode testing."""
        LOG.info(f"[Gov] model {model_id} approved for SHADOW testing")
        self._append_log("shadow_start", {"timestamp": time.time(), "model_id": model_id})

    # ------------------------------------------------------------------
    def shadow_results(self, model_id, replay_stats):
        """
        After replay shadow mode:
        replay_stats: {evr_delta, dd_delta, precision, drift, ...}
        """
        LOG.info(f"[Gov] shadow results for {model_id}: {replay_stats}")

        self._append_log(
            "shadow_results",
            {"timestamp": time.time(), "model_id": model_id, "replay_stats": replay_stats},
        )

        promote = self._shadow_decision(replay_stats)
        return promote

    # ------------------------------------------------------------------
    def _shadow_decision(self, stats: dict):
        cfg = {**self.defaults["shadow_requirements"], **self.config.get("shadow_requirements", {})}
        return bool(
            stats.get("evr_delta", -999) >= cfg.get("min_evr_delta", -999)
            and stats.get("dd_delta", 999) <= cfg.get("max_dd_delta", 999)
            and stats.get("precision", 0.0) >= cfg.get("min_precision", 0.0)
        )

    # ------------------------------------------------------------------
    def promote(self, model_id):
        """Promote model to production."""
        LOG.info(f"[Gov] PROMOTING model {model_id} → PRODUCTION")
        active = None
        if hasattr(self.registry, "set_active_model"):
            active = self.registry.set_active_model(model_id, slot=self.slot)
        self._append_log("promotion", {"timestamp": time.time(), "model_id": model_id, "slot": self.slot, "active": active})
        return active

    # ------------------------------------------------------------------
    def rollback(self, to_model_id):
        """Rollback to a previous production model."""
        LOG.warning(f"[Gov] ROLLBACK → {to_model_id}")
        active = None
        if hasattr(self.registry, "set_active_model"):
            active = self.registry.set_active_model(to_model_id, slot=self.slot)
        self._append_log("rollback", {"timestamp": time.time(), "model_id": to_model_id, "slot": self.slot, "active": active})
        return active

    # ------------------------------------------------------------------
    def history(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        try:
            payload = json.loads(self.log_path.read_text())
            return payload if isinstance(payload, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    def _append_log(self, event, entry):
        log_entry = {"event": event, **entry}
        prev = self.history()
        prev.append(log_entry)
        self.log_path.write_text(json.dumps(prev, indent=2))
