"""
Multi-timeframe SMC structural context:
    - structural bias from 6h
    - premium/discount context
    - BOS/CHOCH recent flags
    - zone scoring metadata
    - compression/expansion flags

Called during feature construction for 15m rows.
"""

import pandas as pd
from typing import Dict, Any

from quant_system.utils.logger import log


class StructureContext:
    """
    Computes SMC structural context at the 6h / 1h level,
    then projects features down to 15m rows.
    """

    def __init__(self, config: Dict[str, Any]):
        smc_cfg = config["features"]["smc"]

        self.discount_th = smc_cfg["premium_discount"]["discount_threshold"]
        self.premium_th = smc_cfg["premium_discount"]["premium_threshold"]
        self.range_ratio_th = smc_cfg["compression"]["range_ratio_threshold"]
        self.vol_drop_th = smc_cfg["compression"]["volatility_drop_pct"]
        self.zone_weights = smc_cfg["zones"]["score_weights"]

        log("StructureContext initialized.")

    # --------------------------------------------------------------
    # Structural bias from 6h BOS / CHOCH
    # --------------------------------------------------------------
    def compute_structural_bias(self, df_6h: pd.DataFrame) -> pd.Series:
        """
        Determines UP / DOWN / NEUTRAL from 6h BOS/CHOCH.
        """
        bias = []
        for i in range(len(df_6h)):
            row = df_6h.iloc[i]

            bos_up = bool(row.get("bos_up", 0))
            bos_down = bool(row.get("bos_down", 0))
            choch_up = bool(row.get("choch_up", 0))
            choch_down = bool(row.get("choch_down", 0))

            if bos_up and not choch_down:
                bias.append("UP")
            elif bos_down and not choch_up:
                bias.append("DOWN")
            else:
                bias.append("NEUTRAL")

        return pd.Series(bias, index=df_6h.index, name="structural_bias_6h")

    # --------------------------------------------------------------
    # Premium / discount context (external range)
    # --------------------------------------------------------------
    def compute_premium_discount(self, df_6h: pd.DataFrame) -> pd.Series:
        """
        Computes PD context: DISCOUNT / PREMIUM / MID.
        """
        pd_vals = []
        for i in range(len(df_6h)):
            row = df_6h.iloc[i]
            pdv = row.get("pd_value", None)

            if pdv is None:
                pd_vals.append("MID")
                continue

            if pdv <= self.discount_th:
                pd_vals.append("DISCOUNT")
            elif pdv >= self.premium_th:
                pd_vals.append("PREMIUM")
            else:
                pd_vals.append("MID")

        return pd.Series(pd_vals, index=df_6h.index, name="pd_context_6h")

    # --------------------------------------------------------------
    # Compression / expansion flags
    # --------------------------------------------------------------
    def compute_compression(self, df_6h: pd.DataFrame) -> pd.Series:
        """
        Compression regime: range_ratio small and volatility drop.
        """
        vals = []
        for i in range(len(df_6h)):
            row = df_6h.iloc[i]
            range_ratio = row.get("range_ratio", None)
            vol_drop = row.get("vol_drop", None)

            if range_ratio is None or vol_drop is None:
                vals.append(0)
                continue

            flag = int(range_ratio <= self.range_ratio_th and vol_drop >= self.vol_drop_th)
            vals.append(flag)

        return pd.Series(vals, index=df_6h.index, name="compression_6h")

    # --------------------------------------------------------------
    # Zone score for 6h zones
    # --------------------------------------------------------------
    def compute_zone_score(self, df_6h: pd.DataFrame) -> pd.Series:
        """
        Weighted zone score based on configured importance.
        """
        w = self.zone_weights
        scores = []

        for i in range(len(df_6h)):
            row = df_6h.iloc[i]

            rec = row.get("zone_recency", 0)
            disp = row.get("zone_displacement", 0)
            mit = row.get("zone_mitigation", 0)
            pd_score = row.get("zone_pd", 0)
            ema = row.get("zone_ema_align", 0)

            score = (
                rec * w["recency"]
                + disp * w["displacement"]
                + mit * w["mitigation"]
                + pd_score * w["premium_discount"]
                + ema * w["ema_alignment"]
            )
            scores.append(score)

        return pd.Series(scores, index=df_6h.index, name="zone_score_6h")

    # --------------------------------------------------------------
    # Projection to 15m via as-of join
    # --------------------------------------------------------------
    def project_to_15m(self, df_15m: pd.DataFrame, df_6h_ctx: pd.DataFrame) -> pd.DataFrame:
        """
        Align 6h context to 15m timestamps using last-known value (asof merge).
        """
        df_6h_ctx = df_6h_ctx.sort_index()
        df_15m = df_15m.sort_index()

        merged = (
            pd.merge_asof(
                df_15m,
                df_6h_ctx,
                left_index=True,
                right_index=True,
                direction="backward",
            )
        )

        log("StructureContext: 6h context projected onto 15m.")
        return merged

    # --------------------------------------------------------------
    # Full pipeline
    # --------------------------------------------------------------
    def build(self, df_6h: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Compute structural context from 6h and merge onto 15m rows.
        """
        bias = self.compute_structural_bias(df_6h)
        pdv = self.compute_premium_discount(df_6h)
        comp = self.compute_compression(df_6h)
        score = self.compute_zone_score(df_6h)

        df_6h_ctx = pd.concat([bias, pdv, comp, score], axis=1)
        out = self.project_to_15m(df_15m, df_6h_ctx)

        log("StructureContext: context features added to 15m frame.")
        return out


class StructureContextBuilder(StructureContext):
    """
    Shim for compatibility with feature builder.
    """
    def apply(self, df_15m: pd.DataFrame, df_6h: pd.DataFrame = None) -> pd.DataFrame:
        """
        Attach 6h structural context onto a 15m dataframe when provided.
        """
        if df_6h is None or df_6h.empty:
            return df_15m

        df_15m = df_15m.copy()
        df_6h = df_6h.copy()

        if "dt" in df_15m.columns:
            df_15m = df_15m.set_index(pd.to_datetime(df_15m["dt"]))
        elif "timestamp" in df_15m.columns:
            df_15m = df_15m.set_index(pd.to_datetime(df_15m["timestamp"], unit="s"))

        if "dt" in df_6h.columns:
            df_6h = df_6h.set_index(pd.to_datetime(df_6h["dt"]))
        elif "timestamp" in df_6h.columns:
            df_6h = df_6h.set_index(pd.to_datetime(df_6h["timestamp"], unit="s"))

        merged = self.build(df_6h, df_15m)
        # dt already present; avoid duplicate column on reset
        merged = merged.reset_index(drop=True)
        return merged
