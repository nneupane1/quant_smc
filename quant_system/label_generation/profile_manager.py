"""Versioned label-profile governance."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from quant_system.utils.logger import console_stage, get_logger

LOG = get_logger("label_profile")

DEFAULT_PROFILE_PATH = Path("artifacts/label_profiles/active_label_profile.json")
DEFAULT_HISTORY_DIR = Path("artifacts/label_profiles/history")


class LabelProfileManager:
    def __init__(
        self,
        *,
        active_profile_path: str | Path = DEFAULT_PROFILE_PATH,
        history_dir: str | Path = DEFAULT_HISTORY_DIR,
    ):
        self.active_profile_path = Path(active_profile_path)
        self.history_dir = Path(history_dir)

    def load_active(self) -> Optional[Dict[str, Any]]:
        if not self.active_profile_path.exists():
            return None
        try:
            payload = json.loads(self.active_profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("[LabelProfileManager] Failed to load active profile: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def resolve_labels_cfg(self, default_cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = deepcopy(default_cfg)
        active = self.load_active()
        if not active:
            return cfg
        tasks = active.get("tasks", {})
        if not isinstance(tasks, dict):
            return cfg
        for task, override in tasks.items():
            if task not in cfg or not isinstance(override, dict):
                continue
            cfg[task].update(override)
        return cfg

    def promote(
        self,
        *,
        tasks: Dict[str, Dict[str, Any]],
        source_summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        now = datetime.now(timezone.utc)
        payload = {
            "promoted_at": now.isoformat(),
            "tasks": deepcopy(tasks),
            "source_summary": deepcopy(source_summary or {}),
        }
        self.active_profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        history_path = self.history_dir / f"label_profile_{stamp}.json"
        history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.active_profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console_stage(
            "Label profile promoted",
            f"active={self.active_profile_path} snapshot={history_path}",
            status="ok",
        )
        return self.active_profile_path
