"""
label_loader.py — full multi-asset label loader.

Generates:
 - BOS continuation (48 bars)
 - Liquidity-flow (12 bars)
 - Micro-momentum (4–8 bars)
 - EOP (96 bars)
 - EDP (96 bars)
 - Hazard survival labels (48 bars)

Inputs:
 - 15m feature frame (already joined with SMC + regime)
 - asset identifier
 - label horizons from labels.yaml
"""

import numpy as np
import pandas as pd
from typing import Dict

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("label_loader")


class LabelLoader:
    """
    Multi-asset aware forward-label generator.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.cfg = config_loader.load_yaml("labels.yaml")
        self.h_bos = self.cfg["horizon"]["bos_cont"]       # e.g. 48
        self.h_liq = self.cfg["horizon"]["liq_flow"]       # e.g. 12
        self.h_momo = self.cfg["horizon"]["momo"]          # e.g. 4
        self.h_eop = self.cfg["horizon"]["eop"]            # e.g. 96
        self.h_edp = self.cfg["horizon"]["edp"]            # e.g. 96
        self.h_haz = self.cfg["horizon"]["hazard"]         # e.g. 48

        LOG.info("[LabelLoader] Initialized with multi-asset capability")

    # ----------------------------------------------------------------------
    # PUBLIC BUILD ENTRYPOINT
    # ----------------------------------------------------------------------
    def build(self, df15: pd.DataFrame, asset: str) -> pd.DataFrame:
        LOG.info(f"[LabelLoader] Building labels for asset={asset}")

        df = df15.copy()
        df = self._bos_cont_label(df)
        df = self._liq_flow_label(df)
        df = self._micro_momo_label(df)
        df = self._eop_label(df)
        df = self._edp_label(df)
        df = self._hazard_label(df)

        df = df.dropna().reset_index(drop=True)
        LOG.info(f"[LabelLoader] Completed labels rows={len(df)} asset={asset}")

        return df

    # ----------------------------------------------------------------------
    # BOS CONTINUATION LABEL 48
    # ----------------------------------------------------------------------
    def _bos_cont_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_bos
        df["label_bos_cont"] = 0

        if "bos_flag_15m" not in df.columns:
            return df

        bos_idx = df.index[df["bos_flag_15m"] == 1].tolist()
        close = df["close"].values

        for i in bos_idx:
            end = min(i + N, len(df) - 1)
            ret = (close[end] - close[i]) / close[i]
            df.loc[i, "label_bos_cont"] = 1 if ret >= 0.03 else 0

        return df

    # ----------------------------------------------------------------------
    # LIQUIDITY FLOW LABEL 12
    # ----------------------------------------------------------------------
    def _liq_flow_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_liq
        df["label_liq_flow"] = 0

        if "sweep_flag_15m" not in df.columns:
            return df

        sw_idx = df.index[df["sweep_flag_15m"] == 1].tolist()
        close = df["close"].values

        for i in sw_idx:
            end = min(i + N, len(df) - 1)
            ret = (close[end] - close[i]) / close[i]
            df.loc[i, "label_liq_flow"] = 1 if ret >= 0.01 else 0

        return df

    # ----------------------------------------------------------------------
    # MICRO MOMENTUM LABEL 4–8
    # ----------------------------------------------------------------------
    def _micro_momo_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_momo
        df["label_momo"] = 0

        close = df["close"].values
        for i in range(len(df)):
            end = min(i + N, len(df) - 1)
            ret = (close[end] - close[i]) / close[i]
            df.loc[i, "label_momo"] = 1 if ret > 0 else 0

        return df

    # ----------------------------------------------------------------------
    # EOP 96  (Expected Opportunity)
    # ----------------------------------------------------------------------
    def _eop_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_eop
        df["label_eop"] = 0

        if "confluence_score" not in df.columns:
            return df

        conf = df["confluence_score"].values

        for i in range(len(df)):
            end = min(i + N, len(df) - 1)
            df.loc[i, "label_eop"] = 1 if conf[i+1:end].max() >= 0.8 else 0

        return df

    # ----------------------------------------------------------------------
    # EDP 96  (Expected Drawdown Probability)
    # ----------------------------------------------------------------------
    def _edp_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_edp
        df["label_edp"] = 0

        close = df["close"].values

        for i in range(len(df)):
            end = min(i + N, len(df) - 1)
            min_ret = (close[i:end].min() - close[i]) / close[i]
            df.loc[i, "label_edp"] = 1 if min_ret <= -0.03 else 0

        return df

    # ----------------------------------------------------------------------
    # HAZARD SURVIVAL LABEL (48 bars)
    # ----------------------------------------------------------------------
    def _hazard_label(self, df: pd.DataFrame) -> pd.DataFrame:
        N = self.h_haz

        df["hazard_event"] = 0
        df["hazard_time"] = N

        close = df["close"].values

        for i in range(len(df)):
            failure_id = None
            for k in range(1, N + 1):
                if i + k >= len(df):
                    break
                ret = (close[i + k] - close[i]) / close[i]
                if ret <= -0.01:
                    failure_id = k
                    break

            if failure_id is not None:
                df.loc[i, "hazard_event"] = 1
                df.loc[i, "hazard_time"] = failure_id
                continue

            df.loc[i, "hazard_event"] = 0
            df.loc[i, "hazard_time"] = N

        return df
