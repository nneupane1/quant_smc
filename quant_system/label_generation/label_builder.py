"""
LabelBuilder
Config-driven label generation using 15m feature frames.

Outputs columns:
  - label_liq_flow
  - label_bos_cont
  - label_momo
  - label_eop
  - label_edp
  - hazard_event (0/1)
  - hazard_time (bars to event or horizon)
"""

from typing import Dict, Any
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("label_builder")


class LabelBuilder:
    def __init__(self, config_loader: ConfigLoader):
        self.cfg_loader = config_loader
        self.labels_cfg = config_loader.load_yaml("labels.yaml")["labels"]

    def apply(self, df15: pd.DataFrame) -> pd.DataFrame:
        """
        Adds label columns to a 15m dataframe (expects close/high/low/atr columns).
        """
        df = df15.copy()
        df = self._liq_flow(df)
        df = self._bos_cont(df)
        df = self._momo(df)
        df = self._eop(df)
        df = self._edp(df)
        df = self._hazard(df)
        return df

    # ------------------------------------------------------------------
    def _atr_col(self, df: pd.DataFrame) -> str:
        for c in ("atr", "atr_15m"):
            if c in df.columns:
                return c
        return None

    def _liq_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["liq_flow"]
        H = int(cfg.get("horizon_bars", 12))
        R_mult = float(cfg.get("continuation_min_R", 1.0))
        atr_col = self._atr_col(df)
        if atr_col is None:
            df["label_liq_flow"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            direction = None
            stop = None
            if row.get("sweep_high", False):
                direction = -1
                stop = row.get("swept_level", row["high"])
            elif row.get("sweep_low", False):
                direction = 1
                stop = row.get("swept_level", row["low"])

            if direction is None or pd.isna(atr):
                labels.append(0)
                continue

            entry = row["close"]
            target = entry + direction * R_mult * atr
            window = df.iloc[i + 1:i + 1 + H]
            success = 0
            for _, r in window.iterrows():
                if direction == 1:
                    if r["low"] <= stop:
                        break
                    if r["high"] >= target:
                        success = 1
                        break
                else:
                    if r["high"] >= stop:
                        break
                    if r["low"] <= target:
                        success = 1
                        break
            labels.append(success)

        df["label_liq_flow"] = labels
        return df

    def _bos_cont(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["bos_cont"]
        H = int(cfg.get("horizon_bars", 48))
        R_mult = float(cfg.get("min_R", 3.0))
        atr_col = self._atr_col(df)
        if atr_col is None:
            df["label_bos_cont"] = 0
            return df

        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            direction = None
            stop = row.get("broken_level", None)
            if row.get("bos_up", False):
                direction = 1
            elif row.get("bos_down", False):
                direction = -1
            if direction is None or stop is None or pd.isna(atr):
                labels.append(0)
                continue

            entry = row["close"]
            target = entry + direction * R_mult * atr
            window = df.iloc[i + 1:i + 1 + H]
            success = 0
            for _, r in window.iterrows():
                if direction == 1:
                    if r["low"] <= stop:
                        break
                    if r["high"] >= target:
                        success = 1
                        break
                else:
                    if r["high"] >= stop:
                        break
                    if r["low"] <= target:
                        success = 1
                        break
            labels.append(success)

        df["label_bos_cont"] = labels
        return df

    def _momo(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["momo"]
        h_min = int(cfg.get("min_horizon", 4))
        h_max = int(cfg.get("max_horizon", 8))
        noise_sigma = float(cfg.get("noise_band_sigma", 0.75))
        ret_sigma = float(cfg.get("return_threshold_sigma", 1.20))

        rets = df["close"].pct_change().fillna(0)
        noise = rets.rolling(50).std().fillna(method="bfill").replace(0, np.nan)
        labels = []
        for i, row in df.iterrows():
            base = row["close"]
            window = df.iloc[i + h_min:i + h_max + 1]
            if window.empty or pd.isna(noise.iloc[i]):
                labels.append(0)
                continue
            fwd_ret = (window["close"] - base) / base
            thr = noise.iloc[i] * ret_sigma + noise_sigma * noise.iloc[i]
            labels.append(int(fwd_ret.max() >= thr))
        df["label_momo"] = labels
        return df

    def _eop(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["eop"]
        H = int(cfg.get("horizon_bars", 96))
        min_evr = float(cfg.get("Aplus_min_evr", 1.5))
        min_medr = float(cfg.get("Aplus_min_medianR", 5.0))
        labels = []
        for i, row in df.iterrows():
            window = df.iloc[i + 1:i + 1 + H]
            ok = False
            if not window.empty:
                ok = ((window.get("evr", pd.Series(0)) >= min_evr) & (window.get("median_r", pd.Series(0)) >= min_medr)).any()
            labels.append(int(ok))
        df["label_eop"] = labels
        return df

    def _edp(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["edp"]
        H = int(cfg.get("horizon_bars", 96))
        dd_r = float(cfg.get("drawdown_R_threshold", -3.0))
        atr_col = self._atr_col(df)
        if atr_col is None:
            df["label_edp"] = 0
            return df
        labels = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            if pd.isna(atr):
                labels.append(0)
                continue
            entry = row["close"]
            stop_move = entry + dd_r * atr
            window = df.iloc[i + 1:i + 1 + H]
            hit = False
            for _, r in window.iterrows():
                if r["low"] <= stop_move:
                    hit = True
                    break
            labels.append(int(hit))
        df["label_edp"] = labels
        return df

    def _hazard(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.labels_cfg["hazard"]
        H = int(cfg.get("horizon_bars", 48))
        event_r = float(cfg.get("event_R_threshold", -1.0))
        atr_col = self._atr_col(df)
        if atr_col is None:
            df["hazard_event"] = 0
            df["hazard_time"] = H
            return df

        events = []
        times = []
        for i, row in df.iterrows():
            atr = row[atr_col]
            if pd.isna(atr):
                events.append(0)
                times.append(H)
                continue
            entry = row["close"]
            stop = entry + event_r * atr
            window = df.iloc[i + 1:i + 1 + H]
            event_hit = 0
            t_hit = H
            for j, (_, r) in enumerate(window.iterrows(), start=1):
                if (r["low"] <= stop and event_r < 0) or (r["high"] >= stop and event_r > 0):
                    event_hit = 1
                    t_hit = j
                    break
            events.append(event_hit)
            times.append(t_hit)

        df["hazard_event"] = events
        df["hazard_time"] = times
        return df


def _cli():
    parser = argparse.ArgumentParser(description="Generate labels from 15m features CSV.")
    parser.add_argument("--config", default="quant_system/config", help="Config directory")
    parser.add_argument("--features", required=True, help="Input features CSV (15m spine with ATR etc.)")
    parser.add_argument("--out", required=True, help="Output labels CSV path")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    lb = LabelBuilder(cfg)

    LOG.info(f"[LabelBuilder] Loading features from {args.features}")
    df = pd.read_csv(args.features)

    LOG.info("[LabelBuilder] Generating labels")
    labels = lb.apply(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    LOG.info(f"[LabelBuilder] Wrote labels to {out_path}")


if __name__ == "__main__":
    _cli()
