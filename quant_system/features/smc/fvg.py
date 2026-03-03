"""
Fair Value Gap (FVG) Detector (leak-safe, with context/decay/broken)
--------------------------------------------------------------------

Classic 3-candle pattern:
    Down FVG: c1.high < c3.low
    Up   FVG: c1.low  > c3.high

Adds:
    - min_gap_pct filter
    - configurable mitigation:
        • require_full_fill=False -> any touch into gap counts as mitigated
        • require_full_fill=True  -> must reach the far boundary
    - Event signals (per bar):
        • fvg_up / fvg_down (creation flags)
        • fvg_gap_top / fvg_gap_bottom / fvg_gap_size
        • fvg_is_broken / fvg_broken_level
        • fvg_open_up / fvg_open_down / fvg_open_total (still-open counts)
    - Context signals (always present, with decay):
        • fvg_ctx_dir ∈ {+1 (up), -1 (down), 0 none}
        • fvg_ctx_age (bars since most recent created FVG)
        • fvg_ctx_fresh / fvg_ctx_stale (thresholded)
        • fvg_ctx_weight = exp(-age/τ) with hard 0 if broken
        • fvg_has_active (1 if any open gap right now)
        • fvg_ctx_dist_top / fvg_ctx_dist_bot (px; normalized to ATR if present)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import math
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


@dataclass
class FVGConfig:
    min_gap_pct: float = 0.0005
    require_full_fill: bool = False
    eps: float = 1e-9
    decay_tau_bars: int = 8
    fresh_bars: int = 8
    stale_bars: int = 24
    carry_context: bool = True


class FVGDetector:
    """
    Detect 3-candle Fair Value Gaps with event + context signals.
    """

    def __init__(
        self,
        min_gap_pct: float = 0.0005,
        require_full_fill: bool = False,
        eps: float = 1e-9,
        decay_tau_bars: int = 8,
        fresh_bars: int = 8,
        stale_bars: int = 24,
        carry_context: bool = True,
    ):
        self.cfg = FVGConfig(
            min_gap_pct=min_gap_pct,
            require_full_fill=require_full_fill,
            eps=eps,
            decay_tau_bars=decay_tau_bars,
            fresh_bars=fresh_bars,
            stale_bars=stale_bars,
            carry_context=carry_context,
        )
        log(
            f"FVGDetector initialized (min_gap_pct={min_gap_pct}, require_full_fill={require_full_fill}, "
            f"tau={decay_tau_bars}, fresh={fresh_bars}, stale={stale_bars})."
        )

    # Optional list API (kept for compatibility; emits event fields only)
    def detect(self, candles: List[Candle]) -> Dict[int, Dict[str, Optional[float]]]:
        log(f"Detecting FVGs for {len(candles):,} candles.")
        out: Dict[int, Dict[str, Optional[float]]] = {}
        if len(candles) < 3:
            return out

        active: List[Dict] = []  # {'top','bottom','dir','age'}
        for i, c3 in enumerate(candles):
            ts = c3.timestamp
            f_up = f_dn = False
            gap_top = gap_bot = gap_size = None

            if i >= 2:
                c1 = candles[i - 2]
                # Up FVG
                if c1.low > c3.high:
                    gap_bot, gap_top = c3.high, c1.low
                    gap_size = gap_top - gap_bot
                    if gap_size / max(self.cfg.eps, c3.close) >= self.cfg.min_gap_pct:
                        f_up = True
                        active.append({"top": gap_top, "bottom": gap_bot, "dir": "up", "age": 0})
                # Down FVG
                if c1.high < c3.low:
                    gap_top, gap_bot = c3.low, c1.high
                    gap_size = gap_top - gap_bot
                    if gap_size / max(self.cfg.eps, c3.close) >= self.cfg.min_gap_pct:
                        f_dn = True
                        active.append({"top": gap_top, "bottom": gap_bot, "dir": "down", "age": 0})

            still_open = []
            for g in active:
                g["age"] += 1
                if not (self._is_mitigated(g, c3) or self._is_broken(g, c3)):
                    still_open.append(g)
            active = still_open

            newest_age = min([g["age"] for g in active], default=np.nan)
            open_up = sum(1 for g in active if g["dir"] == "up")
            open_dn = sum(1 for g in active if g["dir"] == "down")

            out[ts] = {
                "fvg_up": int(f_up),
                "fvg_down": int(f_dn),
                "fvg_gap_top": gap_top,
                "fvg_gap_bottom": gap_bot,
                "fvg_gap_size": gap_size,
                "fvg_age": newest_age,
                "fvg_open_up": open_up,
                "fvg_open_down": open_dn,
                "fvg_open_total": open_up + open_dn,
            }
        return out

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect FVGs and attach per-bar event + persistent context features.
        """
        if df is None or df.empty:
            return df

        frame = df.copy()
        if "timestamp" not in frame.columns:
            if "dt" not in frame.columns:
                raise ValueError("FVGDetector.apply requires 'dt' or 'timestamp' column.")
            frame["timestamp"] = pd.to_datetime(frame["dt"], utc=True).astype("int64") // 10**9

        for col in ("open", "high", "low", "close"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        candles: List[Candle] = [
            Candle(
                timestamp=int(row.timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0)),
            )
            for row in frame.itertuples()
        ]

        records = []
        active: List[Dict] = []  # open gaps
        last_ctx = None  # persistent context of last-created gap

        for i, c in enumerate(candles):
            ts = c.timestamp
            f_up = f_dn = False
            gap_top = gap_bot = gap_size = None
            fvg_is_broken = 0
            fvg_broken_level = np.nan
            fvg_touch_flag = 0
            fvg_filled_flag = 0

            # creation on this bar using c1 and c3 (current)
            if i >= 2:
                c1 = candles[i - 2]
                # Up gap
                if c1.low > c.high:
                    gap_bot, gap_top = c.high, c1.low
                    gap_size = gap_top - gap_bot
                    if gap_size / max(self.cfg.eps, c.close) >= self.cfg.min_gap_pct:
                        f_up = True
                        new_gap = {"top": gap_top, "bottom": gap_bot, "dir": "up", "age": 0, "born_i": i}
                        active.append(new_gap)
                        last_ctx = {"dir": "up", "top": gap_top, "bottom": gap_bot, "born_i": i,
                                    "broken_i": None, "mitigated_i": None}
                # Down gap
                if c1.high < c.low:
                    gap_top, gap_bot = c.low, c1.high
                    gap_size = gap_top - gap_bot
                    if gap_size / max(self.cfg.eps, c.close) >= self.cfg.min_gap_pct:
                        f_dn = True
                        new_gap = {"top": gap_top, "bottom": gap_bot, "dir": "down", "age": 0, "born_i": i}
                        active.append(new_gap)
                        last_ctx = {"dir": "down", "top": gap_top, "bottom": gap_bot, "born_i": i,
                                    "broken_i": None, "mitigated_i": None}

            # resolve open gaps on this close
            still_open = []
            for g in active:
                g["age"] += 1
                mitigated = self._is_mitigated(g, c)
                broken = self._is_broken(g, c)
                touched = self._is_touched(g, c)
                if last_ctx is not None and self._same_gap(g, last_ctx):
                    if broken and last_ctx["broken_i"] is None:
                        last_ctx["broken_i"] = i
                    if mitigated and last_ctx["mitigated_i"] is None:
                        last_ctx["mitigated_i"] = i
                if touched:
                    fvg_touch_flag = 1
                if mitigated:
                    fvg_filled_flag = 1
                if broken and not mitigated:
                    fvg_is_broken = 1
                    fvg_broken_level = c.close
                if not (mitigated or broken):
                    still_open.append(g)
            active = still_open

            open_up = sum(1 for g in active if g["dir"] == "up")
            open_dn = sum(1 for g in active if g["dir"] == "down")
            newest_age_open = min([g["age"] for g in active], default=np.nan)

            # Persistent context
            ctx_dir = 0
            ctx_age = np.nan
            ctx_weight = 0.0
            ctx_fresh = 0
            ctx_stale = 0
            ctx_dist_top = np.nan
            ctx_dist_bot = np.nan
            has_active = int((open_up + open_dn) > 0)

            if self.cfg.carry_context and last_ctx is not None:
                ctx_age = i - last_ctx["born_i"]
                decay = math.exp(-ctx_age / max(1, self.cfg.decay_tau_bars))
                broken_now = last_ctx["broken_i"] is not None and i >= last_ctx["broken_i"]
                ctx_weight = 0.0 if broken_now else float(decay)
                ctx_dir = 1 if last_ctx["dir"] == "up" else -1
                ctx_fresh = int(ctx_age <= self.cfg.fresh_bars)
                ctx_stale = int(ctx_age > self.cfg.stale_bars)
                top, bot = last_ctx["top"], last_ctx["bottom"]
                atr_val = None
                if "atr" in frame.columns:
                    atr_raw = frame["atr"].iloc[i]
                    atr_val = float(atr_raw) if not pd.isna(atr_raw) else None
                if atr_val and atr_val > 0:
                    ctx_dist_top = (top - c.close) / atr_val
                    ctx_dist_bot = (bot - c.close) / atr_val
                else:
                    ctx_dist_top = top - c.close
                    ctx_dist_bot = bot - c.close

            records.append(
                {
                    "timestamp": ts,
                    # event
                    "fvg_up": int(f_up),
                    "fvg_down": int(f_dn),
                    "fvg_gap_top": gap_top,
                    "fvg_gap_bottom": gap_bot,
                    "fvg_gap_size": gap_size,
                    "fvg_mid": (
                        (gap_top + gap_bot) / 2.0
                        if gap_top is not None and gap_bot is not None
                        else np.nan
                    ),
                    "fvg_mid_price": (
                        (gap_top + gap_bot) / 2.0
                        if gap_top is not None and gap_bot is not None
                        else np.nan
                    ),
                    "fvg_hi": gap_top,
                    "fvg_lo": gap_bot,
                    "fvg_is_broken": int(fvg_is_broken),
                    "fvg_broken_level": fvg_broken_level,
                    "fvg_touch_flag": int(fvg_touch_flag),
                    "fvg_filled_flag": int(fvg_filled_flag),
                    "fvg_open_up": open_up,
                    "fvg_open_down": open_dn,
                    "fvg_open_total": open_up + open_dn,
                    "fvg_age": newest_age_open,
                    # context (persistent)
                    "fvg_ctx_dir": ctx_dir,
                    "fvg_ctx_age": ctx_age,
                    "fvg_ctx_weight": ctx_weight,
                    "fvg_ctx_fresh": ctx_fresh,
                    "fvg_ctx_stale": ctx_stale,
                    "fvg_has_active": has_active,
                    "fvg_ctx_dist_top": ctx_dist_top,
                    "fvg_ctx_dist_bot": ctx_dist_bot,
                }
            )

        res = pd.DataFrame(records)
        merged = frame.merge(res, on="timestamp", how="left")
        if "dt" in merged.columns:
            merged = merged.sort_values("dt")
        return merged

    # ---------- helpers ----------
    def _is_mitigated(self, g: Dict, c: Candle) -> bool:
        """Touch- or full-fill mitigation depending on config."""
        if g["dir"] == "up":
            top, bot = g["top"], g["bottom"]
            if self.cfg.require_full_fill:
                return c.low <= (bot + self.cfg.eps)
            return (c.high >= bot - self.cfg.eps) and (c.low <= top + self.cfg.eps)
        else:
            top, bot = g["top"], g["bottom"]
            if self.cfg.require_full_fill:
                return c.high >= (top - self.cfg.eps)
            return (c.low <= top + self.cfg.eps) and (c.high >= bot - self.cfg.eps)

    def _is_broken(self, g: Dict, c: Candle) -> bool:
        """Invalidation before mitigation: close breaches far boundary."""
        if g["dir"] == "up":
            return c.close < (g["bottom"] - self.cfg.eps)
        return c.close > (g["top"] + self.cfg.eps)

    def _is_touched(self, g: Dict, c: Candle) -> bool:
        top, bot = g["top"], g["bottom"]
        return (c.low <= top + self.cfg.eps) and (c.high >= bot - self.cfg.eps)

    def _same_gap(self, g: Dict, ctx: Dict) -> bool:
        return (
            (g["dir"] == ctx["dir"])
            and (abs(g["top"] - ctx["top"]) <= self.cfg.eps)
            and (abs(g["bottom"] - ctx["bottom"]) <= self.cfg.eps)
        )
