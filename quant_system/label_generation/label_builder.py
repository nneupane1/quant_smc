"""Canonical dataframe-first label builder for the 15m execution spine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from quant_system.config.config_loader import ConfigLoader
from quant_system.label_generation.utils import (
    compute_bos_cont_labels,
    compute_edp_labels,
    compute_eop_labels,
    compute_flow_1h_labels,
    compute_hazard_labels,
    compute_liq_flow_labels,
    compute_momo_labels,
)
from quant_system.label_generation.profile_manager import LabelProfileManager
from quant_system.utils.logger import get_logger, runtime_logged

LOG = get_logger("label_builder")


class LabelBuilder:
    LABEL_COLUMNS = [
        "label_liq_flow",
        "label_bos_cont",
        "label_momo",
        "label_flow_1h",
        "label_eop",
        "label_edp",
        "hazard_event",
        "hazard_time",
    ]

    def __init__(self, config_loader: ConfigLoader):
        self.cfg_loader = config_loader
        default_cfg = config_loader.load_yaml("labels.yaml")["labels"]
        self.profile_manager = LabelProfileManager()
        self.labels_cfg = self.profile_manager.resolve_labels_cfg(default_cfg)

    @staticmethod
    def _dataset_signature(df15: pd.DataFrame) -> str:
        if df15 is None or df15.empty:
            return "rows=0"
        first_dt = ""
        last_dt = ""
        if "dt" in df15.columns:
            dt_series = pd.to_datetime(df15["dt"], errors="coerce").dropna()
            if not dt_series.empty:
                first_dt = dt_series.iloc[0].isoformat()
                last_dt = dt_series.iloc[-1].isoformat()
        return f"rows={len(df15)}|first_dt={first_dt}|last_dt={last_dt}"

    @staticmethod
    def _checkpoint_meta_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")

    def _load_checkpoint(
        self,
        checkpoint_path: Optional[Path],
        *,
        dataset_signature: str,
        rows: int,
    ) -> Dict[str, pd.Series]:
        if checkpoint_path is None:
            return {}
        meta_path = self._checkpoint_meta_path(checkpoint_path)
        if not checkpoint_path.exists() or not meta_path.exists():
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return {}
            if str(meta.get("dataset_signature") or "") != str(dataset_signature):
                return {}
            if int(meta.get("rows", -1)) != int(rows):
                return {}
            cdf = pd.read_csv(checkpoint_path)
            if cdf.empty:
                return {}
            if "row_id" not in cdf.columns:
                return {}
            row_id = pd.to_numeric(cdf["row_id"], errors="coerce")
            if row_id.isna().any():
                return {}
            if int(row_id.min()) != 0 or int(row_id.max()) != rows - 1:
                return {}
            out: Dict[str, pd.Series] = {}
            for col in self.LABEL_COLUMNS:
                if col in cdf.columns:
                    out[col] = cdf[col]
            if out:
                LOG.info("[LabelBuilder] Resume checkpoint hit columns=%s path=%s", sorted(out.keys()), checkpoint_path)
            return out
        except Exception as exc:
            LOG.warning("[LabelBuilder] Failed loading checkpoint %s: %s", checkpoint_path, exc)
            return {}

    def _save_checkpoint(
        self,
        checkpoint_path: Optional[Path],
        *,
        dataset_signature: str,
        rows: int,
        df: pd.DataFrame,
    ) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        keep_cols = ["row_id"] + [c for c in self.LABEL_COLUMNS if c in df.columns]
        payload_df = df.reset_index(drop=True).copy()
        payload_df["row_id"] = payload_df.index.astype(int)
        payload_df[keep_cols].to_csv(checkpoint_path, index=False)
        meta = {
            "dataset_signature": dataset_signature,
            "rows": int(rows),
            "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "columns": [c for c in keep_cols if c != "row_id"],
        }
        self._checkpoint_meta_path(checkpoint_path).write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    def _clear_checkpoint(self, checkpoint_path: Optional[Path]) -> None:
        if checkpoint_path is None:
            return
        meta_path = self._checkpoint_meta_path(checkpoint_path)
        checkpoint_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    def apply(
        self,
        df15: pd.DataFrame,
        *,
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        df = df15.copy()
        sig = self._dataset_signature(df)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None

        if resume:
            cached = self._load_checkpoint(ckpt_path, dataset_signature=sig, rows=len(df))
            for col, series in cached.items():
                if len(series) == len(df):
                    df[col] = series.values

        compute_plan = [
            ("label_liq_flow", lambda x: compute_liq_flow_labels(x, self.labels_cfg["liq_flow"])),
            ("label_bos_cont", lambda x: compute_bos_cont_labels(x, self.labels_cfg["bos_cont"])),
            ("label_momo", lambda x: compute_momo_labels(x, self.labels_cfg["momo"])),
            ("label_flow_1h", lambda x: compute_flow_1h_labels(x, self.labels_cfg.get("flow_1h", {}))),
            ("label_eop", lambda x: compute_eop_labels(x, self.labels_cfg["eop"])),
            ("label_edp", lambda x: compute_edp_labels(x, self.labels_cfg["edp"])),
        ]

        for col, fn in compute_plan:
            if col in df.columns:
                continue
            df[col] = fn(df)
            self._save_checkpoint(ckpt_path, dataset_signature=sig, rows=len(df), df=df)

        if "hazard_event" not in df.columns or "hazard_time" not in df.columns:
            hazard_event, hazard_time = compute_hazard_labels(df, self.labels_cfg["hazard"])
            df["hazard_event"] = hazard_event
            df["hazard_time"] = hazard_time
            self._save_checkpoint(ckpt_path, dataset_signature=sig, rows=len(df), df=df)

        self._clear_checkpoint(ckpt_path)
        return df


@runtime_logged("Label builder CLI runtime")
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate labels from a 15m feature CSV.")
    parser.add_argument("--config", default="quant_system/config", help="Config directory")
    parser.add_argument("--features", required=True, help="Input features CSV")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    lb = LabelBuilder(cfg)

    LOG.info("[LabelBuilder] Loading features from %s", args.features)
    df = pd.read_csv(args.features)
    LOG.info("[LabelBuilder] Generating labels")
    labels = lb.apply(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    LOG.info("[LabelBuilder] Wrote labels to %s", out_path)


if __name__ == "__main__":
    _cli()
