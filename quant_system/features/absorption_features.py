"""
Absorption (iceberg) proxy features from bar data.

Uses bar-based signals (no L2 required) to estimate hidden liquidity/absorption.
If spread/OFI data is present, it will boost the score when OFI opposes price.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AbsorptionConfig:
    enable: bool = True
    window_minutes: int = 60
    use_spread_ofi: bool = True
    veto_threshold: float = 0.70
    throttle_band: Tuple[float, float] = (0.40, 0.70)
    cluster_weight: float = 0.4
    ofi_weight: float = 0.15


class AbsorptionFeatureBuilder:
    """
    Iceberg/absorption proxy features (bar-level, no L2 required).
    Inputs expected on 15m spine:
      - 'close','open','high','low','volume' (venue volume)
      - optional 'dollar_volume' (else computed)
      - optional 'ofi' if you ingest spread/orderflow later
    Outputs:
      - absorption_score ∈ [0,1]
      - absorption_near_entry, absorption_near_stop (coarse proximity flags)
    """

    def __init__(self, cfg: Optional[AbsorptionConfig] = None):
        self.cfg = cfg or AbsorptionConfig()

    def apply(self, df15: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.enable or df15.empty:
            return df15

        df = df15.copy()

        # --- ensure numeric inputs ---
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        # Dollar volume fallback
        if "dollar_volume" not in df.columns:
            df["dollar_volume"] = (df["close"] * df["volume"]).astype(float)

        # Rolling window (bars) assuming 15m spine; keep adaptable via window_minutes
        bar_minutes = 15
        w = max(int(self.cfg.window_minutes / bar_minutes), 1)

        # Progress-per-dollar: small price change per $ traded ⇒ absorption
        px_change = (df["close"] - df["open"]).abs().rolling(w, min_periods=w).sum()
        dv_sum = df["dollar_volume"].rolling(w, min_periods=w).sum().replace(0, np.nan)
        progress_per_dollar = (px_change / dv_sum)

        # Robust normalization with expanding fallback
        q_l = progress_per_dollar.rolling(128, min_periods=max(w, 32)).quantile(0.05)
        q_h = progress_per_dollar.rolling(128, min_periods=max(w, 32)).quantile(0.95)
        q_l = q_l.fillna(progress_per_dollar.expanding(min_periods=w).quantile(0.05))
        q_h = q_h.fillna(progress_per_dollar.expanding(min_periods=w).quantile(0.95)).replace(0, 1e-9)
        norm_inv = 1.0 - ((progress_per_dollar - q_l) / (q_h - q_l + 1e-12)).clip(0, 1)

        # Same-price clustering / low-range-high-$ (refresh signature)
        tr = (df["high"] - df["low"]).replace(0, 1e-9)
        cluster_ratio = (df["dollar_volume"] / tr).rolling(w, min_periods=w).median()
        cr_l = cluster_ratio.rolling(128, min_periods=max(w, 32)).quantile(0.05)
        cr_h = cluster_ratio.rolling(128, min_periods=max(w, 32)).quantile(0.95).replace(0, 1e-9)
        cr_l = cr_l.fillna(cluster_ratio.expanding(min_periods=w).quantile(0.05)).fillna(0)
        cr_h = cr_h.fillna(cluster_ratio.expanding(min_periods=w).quantile(0.95))
        cluster_norm = ((cluster_ratio - cr_l) / (cr_h - cr_l + 1e-12)).clip(0, 1)

        # Optional OFI (graded penalty)
        ofi_boost = 0.0
        if self.cfg.use_spread_ofi and "ofi" in df.columns:
            ofi_r = df["ofi"].rolling(w, min_periods=w).mean().fillna(0.0)
            dir_r = np.sign((df["close"] - df["open"]).rolling(w, min_periods=w).sum().fillna(0.0))
            opp = ((dir_r > 0) & (ofi_r <= 0)) | ((dir_r < 0) & (ofi_r >= 0))
            ofi_den = ofi_r.abs().rolling(128, min_periods=max(w, 32)).quantile(0.9).replace(0, 1e-9)
            ofi_mag = (ofi_r.abs() / ofi_den).clip(0, 1)
            ofi_boost = (opp.astype(float) * self.cfg.ofi_weight * ofi_mag)

        # Fuse score
        w_cluster = self.cfg.cluster_weight
        score = ((1.0 - w_cluster) * norm_inv + w_cluster * cluster_norm + ofi_boost).clip(0, 1)
        df["absorption_score"] = score.fillna(0.0)

        # Coarse proximity flags; upgrade later with zone distances
        df["absorption_near_entry"] = (cluster_norm > 0.65).fillna(False).astype(int)
        df["absorption_near_stop"] = (norm_inv > 0.65).fillna(False).astype(int)

        return df
