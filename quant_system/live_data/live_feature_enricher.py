"""Runtime multi-timeframe feature enrichment for live 15m bars."""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant_system.live_data.quote_state import QuoteState
from quant_system.utils.logger import get_logger

LOG = get_logger("live_feature_enricher")


def _as_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


def _parse_clock(s: str, fallback: str) -> time:
    text = str(s or fallback)
    hh, mm = text.split(":", 1)
    return time(hour=int(hh), minute=int(mm))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


class LiveFeatureEnricher:
    """Builds execution-critical 15m/1h/6h/12h features from live bar state."""

    def __init__(self, config: Any):
        cfg = _as_dict(config)
        exec_cfg = cfg.get("execution", {})
        gates_cfg = exec_cfg.get("gates", {})
        feat_cfg = cfg.get("features", {})
        vol_cfg = feat_cfg.get("volatility", {})
        session_cfg = feat_cfg.get("session", {})

        self.strict_mode = bool(gates_cfg.get("strict_mode", False))
        self.flow_cfg = gates_cfg.get("flow_1h", {})
        self.atr_period = int(vol_cfg.get("atr_period", 14))
        self.vol_z_window = int(vol_cfg.get("volatility_z_window", 200))
        self.tz = ZoneInfo(str(session_cfg.get("timezone", "Europe/Berlin")))

        london_cfg = session_cfg.get("london", {})
        ny_cfg = session_cfg.get("ny", {})
        overlap_cfg = session_cfg.get("overlap", {})
        off_cfg = session_cfg.get("off_hours", {})
        self.london_start = _parse_clock(london_cfg.get("start"), "08:00")
        self.london_end = _parse_clock(london_cfg.get("end"), "17:00")
        self.ny_start = _parse_clock(ny_cfg.get("start"), "14:30")
        self.ny_end = _parse_clock(ny_cfg.get("end"), "21:00")
        self.weight_london = float(london_cfg.get("weight", 1.0))
        self.weight_ny = float(ny_cfg.get("weight", 1.0))
        self.weight_overlap = float(overlap_cfg.get("weight", 1.1))
        self.weight_off = float(off_cfg.get("weight", 0.2))

    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out
        if "dt" not in out.columns:
            if "timestamp" in out.columns:
                out["dt"] = pd.to_datetime(out["timestamp"], unit="s", utc=True, errors="coerce")
            else:
                out["dt"] = pd.NaT
        else:
            out["dt"] = pd.to_datetime(out["dt"], utc=True, errors="coerce")
        return out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    @staticmethod
    def _ohlcv_resample(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df_1m.empty:
            return pd.DataFrame()
        df = df_1m.copy()
        df = df.set_index("dt")
        agg = df.resample(rule, label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "asset": "last",
            }
        )
        agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
        if agg.empty:
            return agg
        agg["timestamp"] = (agg["dt"].astype("int64") // 10**9).astype(int)
        return agg

    def _window(self, state: QuoteState, tf: str) -> pd.DataFrame:
        direct = self._ensure_dt(state.get_window(tf))
        if not direct.empty:
            return direct

        one_min = self._ensure_dt(state.get_window("1m"))
        if one_min.empty:
            return pd.DataFrame()

        if tf == "15m":
            return self._ohlcv_resample(one_min, "15min")
        if tf == "1h":
            return self._ohlcv_resample(one_min, "1h")
        if tf == "6h":
            return self._ohlcv_resample(one_min, "6h")
        if tf == "12h":
            return self._ohlcv_resample(one_min, "12h")
        return pd.DataFrame()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(max(int(period), 1), min_periods=1).mean()

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        w = max(int(window), 2)
        m = series.rolling(w, min_periods=2).mean()
        s = series.rolling(w, min_periods=2).std(ddof=0).replace(0.0, np.nan)
        return ((series - m) / s).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=max(int(span), 2), adjust=False).mean()

    @staticmethod
    def _in_window(t: time, start: time, end: time) -> bool:
        return start <= t < end

    def _session(self, dt_utc: pd.Timestamp) -> Dict[str, Any]:
        local = dt_utc.tz_convert(self.tz)
        tod = local.time()
        in_london = self._in_window(tod, self.london_start, self.london_end)
        in_ny = self._in_window(tod, self.ny_start, self.ny_end)
        in_overlap = in_london and in_ny
        off = int(not (in_london or in_ny))
        if in_overlap:
            weight = self.weight_overlap
            session = "overlap"
        elif in_london:
            weight = self.weight_london
            session = "london"
        elif in_ny:
            weight = self.weight_ny
            session = "ny"
        else:
            weight = self.weight_off
            session = "off_hours"
        return {
            "session_london": int(in_london),
            "session_ny": int(in_ny),
            "session_overlap": int(in_overlap),
            "session_offhours": off,
            "session_off_hours": off,
            "session_weight": float(weight),
            "is_ldn": int(in_london),
            "is_ny": int(in_ny),
            "session": session,
            "session_name": session,
        }

    @staticmethod
    def _normalize_probs(raw: Dict[str, float]) -> Dict[str, float]:
        vals = {k: max(float(v), 1e-6) for k, v in raw.items()}
        s = sum(vals.values())
        if s <= 0:
            n = 1.0 / max(len(vals), 1)
            return {k: n for k in vals}
        return {k: v / s for k, v in vals.items()}

    def enrich(self, state: QuoteState, asset: str, bar_15m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df15 = self._window(state, "15m")
        df1h = self._window(state, "1h")
        df6h = self._window(state, "6h")
        df12h = self._window(state, "12h")

        if df15.empty:
            if self.strict_mode:
                LOG.warning("[LiveFeatureEnricher] strict drop for %s: 15m window empty", asset)
                return None
            out = dict(bar_15m)
            out["asset"] = asset
            return out

        close15 = pd.to_numeric(df15["close"], errors="coerce")
        open15 = pd.to_numeric(df15["open"], errors="coerce")
        high15 = pd.to_numeric(df15["high"], errors="coerce")
        low15 = pd.to_numeric(df15["low"], errors="coerce")
        atr15 = self._atr(df15, self.atr_period)
        range_pct15 = ((high15 - low15) / close15.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol_z15 = self._zscore(range_pct15, self.vol_z_window)

        ema21 = self._ema(close15, 21)
        ema55 = self._ema(close15, 55)
        dist_ema = ((close15 - ema21) / close15.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ema_dist_z = self._zscore(dist_ema, 120)
        body_pct_15m = ((close15 - open15).abs() / (high15 - low15).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        flow_body = 0.0
        flow_vol_z = 0.0
        flow_signal = 0
        flow_age = 999
        p_flow = 0.50
        if not df1h.empty:
            close1h = pd.to_numeric(df1h["close"], errors="coerce")
            open1h = pd.to_numeric(df1h["open"], errors="coerce")
            high1h = pd.to_numeric(df1h["high"], errors="coerce")
            low1h = pd.to_numeric(df1h["low"], errors="coerce")
            vol1h = pd.to_numeric(df1h["volume"], errors="coerce")

            body1h = ((close1h - open1h).abs() / (high1h - low1h).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            vol_z_1h = self._zscore(vol1h, 60)
            threshold_body = float(self.flow_cfg.get("displacement_body_pct_min", 0.60))
            threshold_vol = float(self.flow_cfg.get("volume_z_min", 0.80))
            flow_mask = (body1h >= threshold_body) & (vol_z_1h >= threshold_vol)
            flow_idx = np.flatnonzero(flow_mask.values)
            flow_age = int((len(body1h) - 1) - flow_idx[-1]) if len(flow_idx) > 0 else 999
            flow_body = _safe_float(body1h.iloc[-1], 0.0)
            flow_vol_z = _safe_float(vol_z_1h.iloc[-1], 0.0)
            flow_signal = int(bool(flow_mask.iloc[-1])) if len(flow_mask) > 0 else 0
            flow_strength = np.clip(0.65 * flow_body + 0.35 * max(flow_vol_z, 0.0) / 3.0, 0.0, 1.5)
            p_flow = float(np.clip(0.15 + 0.7 * flow_strength, 0.01, 0.99))

        side = "long"
        structural_bias_6h = "up"
        zone_score_6h = 0.50
        if not df6h.empty:
            close6 = pd.to_numeric(df6h["close"], errors="coerce")
            ema6_fast = self._ema(close6, 8)
            ema6_slow = self._ema(close6, 21)
            slope6 = _safe_float(ema6_fast.diff().iloc[-1], 0.0)
            spread6 = _safe_float(((ema6_fast - ema6_slow) / close6.replace(0.0, np.nan)).iloc[-1], 0.0)
            structural_bias_6h = "up" if (spread6 >= 0 and slope6 >= 0) else "down"
            side = "long" if structural_bias_6h == "up" else "short"
            trend_consistency = float((close6.diff().fillna(0.0).tail(12) > 0).mean())
            if structural_bias_6h == "down":
                trend_consistency = float((close6.diff().fillna(0.0).tail(12) < 0).mean())
            zone_score_6h = float(np.clip(0.35 + 0.45 * abs(spread6) * 100.0 + 0.20 * trend_consistency, 0.0, 1.0))

        p_trend_up = 0.55 if side == "long" else 0.45
        p_trend_down = 1.0 - p_trend_up
        p_regime_trend = 0.50
        p_regime_expansion = 0.25
        p_regime_collapse = 0.10
        p_regime_range = 0.15
        toxicity_12h = 0.20
        compression_12h = 0.40
        if not df12h.empty:
            close12 = pd.to_numeric(df12h["close"], errors="coerce")
            high12 = pd.to_numeric(df12h["high"], errors="coerce")
            low12 = pd.to_numeric(df12h["low"], errors="coerce")
            ret12 = close12.pct_change().fillna(0.0)
            atr12 = self._atr(df12h, self.atr_period)
            ema12_fast = self._ema(close12, 12)
            ema12_slow = self._ema(close12, 48)

            spread12 = _safe_float(((ema12_fast - ema12_slow) / close12.replace(0.0, np.nan)).iloc[-1], 0.0)
            trend_strength = float(np.clip(0.5 + 6.0 * spread12, 0.01, 0.99))
            p_trend_up = trend_strength
            p_trend_down = float(np.clip(1.0 - trend_strength, 0.01, 0.99))
            side = "long" if p_trend_up >= p_trend_down else "short"

            atr_pct = float(atr12.rank(pct=True).iloc[-1]) if len(atr12) else 0.5
            range12 = ((high12 - low12) / close12.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            compression_12h = float(np.clip(1.0 - range12.rolling(10, min_periods=2).mean().iloc[-1] * 20.0, 0.0, 1.0))
            tox_raw = ret12.abs().rolling(16, min_periods=2).mean()
            tox_scale = tox_raw.rank(pct=True).iloc[-1] if len(tox_raw) else 0.2
            toxicity_12h = float(np.clip(float(tox_scale), 0.0, 1.0))

            p_regime_trend = float(np.clip(0.20 + 0.70 * abs(p_trend_up - p_trend_down), 0.01, 0.99))
            p_regime_expansion = float(np.clip(0.10 + 0.70 * atr_pct * (1.0 - compression_12h), 0.01, 0.99))
            p_regime_collapse = float(np.clip(0.05 + 0.60 * compression_12h * (1.0 - atr_pct), 0.01, 0.99))
            raw = self._normalize_probs(
                {
                    "trend": p_regime_trend,
                    "expansion": p_regime_expansion,
                    "collapse": p_regime_collapse,
                    "range": 0.20 + 0.4 * compression_12h,
                }
            )
            p_regime_trend = raw["trend"]
            p_regime_expansion = raw["expansion"]
            p_regime_collapse = raw["collapse"]
            p_regime_range = raw["range"]

        dt = pd.to_datetime(bar_15m.get("dt"), utc=True, errors="coerce")
        if pd.isna(dt):
            dt = pd.Timestamp.now(tz="UTC")
        session = self._session(dt)

        atr_now = max(_safe_float(atr15.iloc[-1], 0.0), 1e-9)
        close_now = _safe_float(bar_15m.get("close"), _safe_float(close15.iloc[-1], 0.0))
        if side == "long":
            swing_target = close_now + 2.0 * atr_now
            fvg_target = close_now + 3.0 * atr_now
            liquidity_pool = close_now + 4.0 * atr_now
            range_edge = close_now + 1.5 * atr_now
        else:
            swing_target = close_now - 2.0 * atr_now
            fvg_target = close_now - 3.0 * atr_now
            liquidity_pool = close_now - 4.0 * atr_now
            range_edge = close_now - 1.5 * atr_now

        flow_fresh_max = int(self.flow_cfg.get("freshness_bars", 4))
        flow_ok = bool(flow_body >= float(self.flow_cfg.get("displacement_body_pct_min", 0.60)) and flow_vol_z >= float(self.flow_cfg.get("volume_z_min", 0.80)) and flow_age <= flow_fresh_max)

        regime_probs = {
            "trend": p_regime_trend,
            "expansion": p_regime_expansion,
            "collapse": p_regime_collapse,
            "range": p_regime_range,
        }
        regime_state = max(regime_probs, key=regime_probs.get)
        regime_state_id = {"trend": 0, "expansion": 1, "collapse": 2, "range": 3}[regime_state]

        enriched = dict(bar_15m)
        enriched.update(
            {
                "asset": asset,
                "dt": dt.to_pydatetime(),
                "timestamp": int(dt.timestamp()),
                "side": side,
                "atr": _safe_float(atr15.iloc[-1], 0.0),
                "atr_15m": _safe_float(atr15.iloc[-1], 0.0),
                "vol_zscore": _safe_float(vol_z15.iloc[-1], 0.0),
                "range_pct": _safe_float(range_pct15.iloc[-1], 0.0),
                "body_pct_15m": _safe_float(body_pct_15m.iloc[-1], 0.0),
                "ema_dist_z": _safe_float(ema_dist_z.iloc[-1], 0.0),
                "flow_signal_1h": int(flow_signal),
                "displacement_body_pct_1h": float(flow_body),
                "volume_z_1h": float(flow_vol_z),
                "flow_age_bars_1h": int(flow_age),
                "flow_ok_1h": int(flow_ok),
                "p_flow_1h": float(p_flow),
                "prob_flow_1h": float(p_flow),
                "structural_bias_6h": structural_bias_6h,
                "structure_bias_6h": structural_bias_6h,
                "zone_score_6h": float(zone_score_6h),
                "p_trend_up_12h": float(p_trend_up),
                "p_trend_down_12h": float(p_trend_down),
                "p_trend_up": float(p_trend_up),
                "p_trend_down": float(p_trend_down),
                "toxicity_12h": float(toxicity_12h),
                "compression_12h": float(compression_12h),
                "p_regime_trend": float(p_regime_trend),
                "p_regime_expansion": float(p_regime_expansion),
                "p_regime_collapse": float(p_regime_collapse),
                "p_regime_range": float(p_regime_range),
                "regime_state": regime_state,
                "regime_state_id": int(regime_state_id),
                "swing_target": float(swing_target),
                "fvg_target": float(fvg_target),
                "liquidity_pool": float(liquidity_pool),
                "range_edge": float(range_edge),
            }
        )
        enriched.update(session)

        return enriched
