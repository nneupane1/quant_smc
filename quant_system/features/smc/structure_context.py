"""
Multi-timeframe SMC structural context:
    - structural bias from 6h
    - premium/discount context
    - BOS/CHOCH recent flags
    - zone scoring metadata
    - compression/expansion flags

Called during feature construction for 15m rows.
"""

import numpy as np
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
        self.external_range_lookback = int(smc_cfg["premium_discount"].get("external_range_lookback", 150))
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
            legacy_bias = row.get("bias", row.get("structure_bias"))
            demand_q = row.get("demand_quality", None)
            supply_q = row.get("supply_quality", None)

            if bos_up and not choch_down:
                bias.append("UP")
            elif bos_down and not choch_up:
                bias.append("DOWN")
            elif isinstance(legacy_bias, str) and legacy_bias:
                bias.append(legacy_bias.upper())
            elif demand_q is not None and supply_q is not None:
                bias.append("UP" if float(demand_q) >= float(supply_q) else "DOWN")
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
        lookback = max(3, min(self.external_range_lookback, max(len(df_6h), 3)))
        min_periods = max(1, min(5, lookback))
        rolling_high = df_6h["high"].rolling(lookback, min_periods=min_periods).max()
        rolling_low = df_6h["low"].rolling(lookback, min_periods=min_periods).min()
        for i in range(len(df_6h)):
            row = df_6h.iloc[i]
            pdv = row.get("pd_value", None)
            if pdv is None:
                ext_hi = rolling_high.iloc[i]
                ext_lo = rolling_low.iloc[i]
                rng = ext_hi - ext_lo
                if pd.notna(rng) and rng > 0:
                    pdv = (row.get("close", np.nan) - ext_lo) / rng

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
        range_pct = ((df_6h["high"] - df_6h["low"]) / df_6h["close"].replace(0, pd.NA)).astype(float)
        lookback = min(20, max(len(df_6h), 1))
        min_periods = max(1, min(5, lookback))
        range_roll = range_pct.rolling(lookback, min_periods=min_periods).mean()
        vol_drop_series = ((range_roll.shift(5) - range_roll) / range_roll.shift(5).replace(0, pd.NA)).fillna(0.0)
        for i in range(len(df_6h)):
            row = df_6h.iloc[i]
            range_ratio = row.get("range_ratio", row.get("range_pct", range_pct.iloc[i] if len(range_pct) > i else None))
            vol_drop = row.get("vol_drop", vol_drop_series.iloc[i] if len(vol_drop_series) > i else None)

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
            disp = row.get("zone_displacement", row.get("demand_quality", row.get("supply_quality", 0)))
            mit = row.get("zone_mitigation", 1 - max(float(row.get("demand_touched", 0) or 0), float(row.get("supply_touched", 0) or 0)))
            pd_score = row.get("zone_pd", 0)
            ema = row.get("zone_ema_align", 0)

            if not rec:
                zone_age = row.get("demand_age", row.get("supply_age"))
                if zone_age is not None:
                    rec = 1.0 / (1.0 + max(float(zone_age), 0.0))
            if not pd_score:
                pdv = row.get("pd_value")
                if pdv is not None and pd.notna(pdv):
                    if pdv <= self.discount_th:
                        pd_score = 1.0
                    elif pdv >= self.premium_th:
                        pd_score = 0.0
                    else:
                        pd_score = 0.5

            score = (
                float(rec) * w["recency"]
                + float(disp) * w["displacement"]
                + float(mit) * w["mitigation"]
                + float(pd_score) * w["premium_discount"]
                + float(ema) * w["ema_alignment"]
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
        df_6h_ctx["bias_6h"] = df_6h_ctx["structural_bias_6h"]
        df_6h_ctx["structure_bias_6h"] = df_6h_ctx["structural_bias_6h"]
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
