"""Shared dataframe-first label helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from quant_system.data.store.datamodel import Candle


def frame_from_candles(candles: Iterable[Candle]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": int(c.timestamp),
            "dt": pd.to_datetime(int(c.timestamp), unit="s", utc=True).tz_convert(None),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        }
        for c in candles
    ]
    return pd.DataFrame(rows)


def map_series(df: pd.DataFrame, values: Dict[Any, Any], name: str) -> pd.Series:
    if "timestamp" in df.columns:
        key = df["timestamp"]
    elif "dt" in df.columns:
        key = pd.to_datetime(df["dt"], errors="coerce").astype("int64") // 10**9
    else:
        key = pd.Series(range(len(df)), index=df.index)
    return key.map(values).rename(name)


def resolve_atr_col(df: pd.DataFrame) -> str | None:
    for col in ("atr_15m", "atr"):
        if col in df.columns:
            return col
    return None


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "up", "down", "long", "short"}
    if pd.isna(value):
        return False
    return bool(value)


def _numeric(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _string(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().upper()


def infer_direction(row: Dict[str, Any]) -> int:
    side = _string(row.get("side"))
    if side == "LONG":
        return 1
    if side == "SHORT":
        return -1

    flow_signal = int(_numeric(row.get("flow_signal_1h"), 0.0))
    if flow_signal > 0:
        return 1
    if flow_signal < 0:
        return -1

    sweep_dir = int(_numeric(row.get("sweep_dir"), 0.0))
    if sweep_dir != 0:
        return 1 if sweep_dir > 0 else -1
    if _boolish(row.get("sweep_low")):
        return 1
    if _boolish(row.get("sweep_high")):
        return -1

    if _boolish(row.get("bos_up")):
        return 1
    if _boolish(row.get("bos_down")):
        return -1

    bias = _string(row.get("bias") or row.get("structure_bias") or row.get("structural_bias_6h"))
    if bias == "UP":
        return 1
    if bias == "DOWN":
        return -1
    return 0


def opposite_choch(row: Dict[str, Any], direction: int) -> bool:
    if direction > 0:
        if _boolish(row.get("choch_down")):
            return True
        return _boolish(row.get("choch_flag")) and _string(row.get("bias") or row.get("structure_bias")) == "DOWN"
    if direction < 0:
        if _boolish(row.get("choch_up")):
            return True
        return _boolish(row.get("choch_flag")) and _string(row.get("bias") or row.get("structure_bias")) == "UP"
    return False


def confluence_series(df: pd.DataFrame) -> pd.Series:
    for col in ("confluence_score", "conf_score", "prob_confluence"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def hazard_series(df: pd.DataFrame) -> pd.Series:
    for col in ("hazard_score", "hazard"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _bar_body_pct(row: Dict[str, Any]) -> float:
    high = _numeric(row.get("high"))
    low = _numeric(row.get("low"))
    open_ = _numeric(row.get("open"))
    close = _numeric(row.get("close"))
    rng = max(high - low, 1e-9)
    return abs(close - open_) / rng


def compute_liq_flow_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    atr_col = resolve_atr_col(df15)
    if atr_col is None:
        return pd.Series(0, index=df15.index, dtype=int, name="label_liq_flow")

    H = int(cfg.get("horizon_bars", 12))
    R_mult = float(cfg.get("continuation_min_R", 1.0))
    min_body = float(cfg.get("displacement_min_body_pct", 0.0))
    min_vol_z = float(cfg.get("volume_z_threshold", 0.0))
    require_sweep = bool(cfg.get("smc_requirements", {}).get("require_sweep", True))
    require_displacement = bool(cfg.get("smc_requirements", {}).get("require_displacement", False))

    labels: List[int] = []
    for i, row in df15.iterrows():
        direction = 0
        if _boolish(row.get("sweep_low")):
            direction = 1
        elif _boolish(row.get("sweep_high")):
            direction = -1
        elif _boolish(row.get("sweep_flag")):
            direction = 1 if _numeric(row.get("sweep_dir"), 0.0) >= 0 else -1

        if require_sweep and direction == 0:
            labels.append(0)
            continue

        atr = _numeric(row.get(atr_col), np.nan)
        if pd.isna(atr) or atr <= 0:
            labels.append(0)
            continue

        if require_displacement:
            body_pct = _numeric(row.get("displacement_body_pct_15m"), _numeric(row.get("displacement_hint"), _bar_body_pct(row)))
            vol_z = _numeric(row.get("volume_z_15m"), _numeric(row.get("volume_z"), min_vol_z))
            if body_pct < min_body or vol_z < min_vol_z:
                labels.append(0)
                continue

        entry = _numeric(row.get("close"))
        stop = row.get("swept_level")
        if stop is None or pd.isna(stop):
            stop = _numeric(row.get("low")) if direction > 0 else _numeric(row.get("high"))
        else:
            stop = _numeric(stop)
        target = entry + direction * R_mult * atr
        success = 0
        for _, future in df15.iloc[i + 1:i + 1 + H].iterrows():
            if direction > 0:
                if _numeric(future.get("low")) <= stop:
                    break
                if _numeric(future.get("high")) >= target:
                    success = 1
                    break
            else:
                if _numeric(future.get("high")) >= stop:
                    break
                if _numeric(future.get("low")) <= target:
                    success = 1
                    break
        labels.append(success)

    return pd.Series(labels, index=df15.index, dtype=int, name="label_liq_flow")


def compute_bos_cont_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    atr_col = resolve_atr_col(df15)
    if atr_col is None:
        return pd.Series(0, index=df15.index, dtype=int, name="label_bos_cont")

    H = int(cfg.get("horizon_bars", 48))
    reward_r = float(cfg.get("min_R", 3.0))
    atr_buffer = float(cfg.get("bos_atr_buffer_mult", 0.0))
    invalidation_on_choch = bool(cfg.get("invalidation_on_choch", True))

    labels: List[int] = []
    for i, row in df15.iterrows():
        direction = 0
        if _boolish(row.get("bos_up")):
            direction = 1
        elif _boolish(row.get("bos_down")):
            direction = -1
        elif _boolish(row.get("bos_flag")):
            direction = infer_direction(dict(row))

        atr = _numeric(row.get(atr_col), np.nan)
        if direction == 0 or pd.isna(atr) or atr <= 0:
            labels.append(0)
            continue

        entry = _numeric(row.get("close"))
        if direction > 0:
            base_stop = row.get("broken_level", row.get("zone_lo", row.get("zone_low")))
            stop = _numeric(base_stop, entry - atr)
            stop -= atr_buffer * atr
        else:
            base_stop = row.get("broken_level", row.get("zone_hi", row.get("zone_high")))
            stop = _numeric(base_stop, entry + atr)
            stop += atr_buffer * atr
        target = entry + direction * reward_r * atr

        success = 0
        for _, future in df15.iloc[i + 1:i + 1 + H].iterrows():
            if invalidation_on_choch and opposite_choch(dict(future), direction):
                break
            if direction > 0:
                if _numeric(future.get("low")) <= stop:
                    break
                if _numeric(future.get("high")) >= target:
                    success = 1
                    break
            else:
                if _numeric(future.get("high")) >= stop:
                    break
                if _numeric(future.get("low")) <= target:
                    success = 1
                    break
        labels.append(success)

    return pd.Series(labels, index=df15.index, dtype=int, name="label_bos_cont")


def compute_momo_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    h_min = int(cfg.get("min_horizon", 4))
    h_max = int(cfg.get("max_horizon", 8))
    noise_sigma = float(cfg.get("noise_band_sigma", 0.75))
    ret_sigma = float(cfg.get("return_threshold_sigma", 1.20))

    rets = pd.to_numeric(df15["close"], errors="coerce").pct_change().fillna(0.0)
    noise = rets.rolling(50).std().bfill().replace(0, np.nan)
    labels: List[int] = []
    for i, row in df15.iterrows():
        base = _numeric(row.get("close"))
        window = df15.iloc[i + h_min:i + h_max + 1]
        if base <= 0 or window.empty or pd.isna(noise.iloc[i]):
            labels.append(0)
            continue
        fwd_ret = (pd.to_numeric(window["close"], errors="coerce") - base) / base
        thr = noise.iloc[i] * (ret_sigma + noise_sigma)
        labels.append(int(fwd_ret.max() >= thr))
    return pd.Series(labels, index=df15.index, dtype=int, name="label_momo")


def compute_flow_1h_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    atr_col = resolve_atr_col(df15)
    if atr_col is None:
        return pd.Series(0, index=df15.index, dtype=int, name="label_flow_1h")

    h_min = int(cfg.get("min_horizon", 4))
    h_max = int(cfg.get("max_horizon", 12))
    min_r = float(cfg.get("min_R", 1.5))
    stop_r = float(cfg.get("stop_R", 1.0))
    min_body = float(cfg.get("min_displacement_body_pct", 0.60))
    min_vol_z = float(cfg.get("min_volume_z", 0.80))
    max_age = int(cfg.get("max_flow_age_bars", 4))

    labels: List[int] = []
    for i, row in df15.iterrows():
        flow_signal = int(_numeric(row.get("flow_signal_1h"), 0.0))
        body_pct = _numeric(row.get("displacement_body_pct_1h"), 0.0)
        volume_z = _numeric(row.get("volume_z_1h"), 0.0)
        flow_age = int(_numeric(row.get("flow_age_bars_1h"), 999.0))
        atr = _numeric(row.get(atr_col), np.nan)
        if flow_signal == 0 or body_pct < min_body or volume_z < min_vol_z or flow_age > max_age or pd.isna(atr) or atr <= 0:
            labels.append(0)
            continue

        direction = 1 if flow_signal > 0 else -1
        entry = _numeric(row.get("close"))
        target = entry + direction * min_r * atr
        stop = entry - direction * stop_r * atr
        success = 0
        for _, future in df15.iloc[i + h_min:i + h_max + 1].iterrows():
            if direction > 0:
                if _numeric(future.get("low")) <= stop:
                    break
                if _numeric(future.get("high")) >= target:
                    success = 1
                    break
            else:
                if _numeric(future.get("high")) >= stop:
                    break
                if _numeric(future.get("low")) <= target:
                    success = 1
                    break
        labels.append(success)

    return pd.Series(labels, index=df15.index, dtype=int, name="label_flow_1h")


def compute_eop_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    H = int(cfg.get("horizon_bars", 96))
    conf_min = float(cfg.get("Aplus_min_conf", 0.80))
    evr_min = float(cfg.get("Aplus_min_evr", 1.5))
    medr_min = float(cfg.get("Aplus_min_medianR", 5.0))
    haz_cap = float(cfg.get("hazard_cap", 0.35))

    conf = confluence_series(df15)
    hazard = hazard_series(df15)
    evr = pd.to_numeric(df15.get("evr", pd.Series(0.0, index=df15.index)), errors="coerce").fillna(0.0)
    medr = pd.to_numeric(df15.get("median_r", pd.Series(0.0, index=df15.index)), errors="coerce").fillna(0.0)
    tier = df15.get("tier", pd.Series("", index=df15.index)).astype(str).str.upper()

    labels: List[int] = []
    for i in range(len(df15)):
        window = slice(i + 1, i + 1 + H)
        future_a_plus = tier.iloc[window].eq("A+")
        future_ok = (conf.iloc[window] >= conf_min) | future_a_plus
        future_ok &= evr.iloc[window] >= evr_min
        future_ok &= medr.iloc[window] >= medr_min
        future_ok &= hazard.iloc[window] <= haz_cap
        labels.append(int(bool(future_ok.any())))
    return pd.Series(labels, index=df15.index, dtype=int, name="label_eop")


def compute_edp_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    atr_col = resolve_atr_col(df15)
    if atr_col is None:
        return pd.Series(0, index=df15.index, dtype=int, name="label_edp")

    H = int(cfg.get("horizon_bars", 96))
    dd_r = abs(float(cfg.get("drawdown_R_threshold", -3.0)))
    labels: List[int] = []
    for i, row in df15.iterrows():
        atr = _numeric(row.get(atr_col), np.nan)
        entry = _numeric(row.get("close"))
        if pd.isna(atr) or atr <= 0 or entry <= 0:
            labels.append(0)
            continue
        target = entry - dd_r * atr
        future = df15.iloc[i + 1:i + 1 + H]
        labels.append(int(bool((pd.to_numeric(future.get("low", pd.Series(dtype=float)), errors="coerce") <= target).any())))
    return pd.Series(labels, index=df15.index, dtype=int, name="label_edp")


def compute_hazard_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    atr_col = resolve_atr_col(df15)
    H = int(cfg.get("horizon_bars", 48))
    if atr_col is None:
        return (
            pd.Series(0, index=df15.index, dtype=int, name="hazard_event"),
            pd.Series(H, index=df15.index, dtype=int, name="hazard_time"),
        )

    event_r = abs(float(cfg.get("event_R_threshold", -1.0)))
    treat_choch = bool(cfg.get("treat_choch_as_event", True))
    treat_stop = bool(cfg.get("treat_stop_as_event", True))

    events: List[int] = []
    times: List[int] = []
    for i, row in df15.iterrows():
        atr = _numeric(row.get(atr_col), np.nan)
        entry = _numeric(row.get("close"))
        direction = infer_direction(dict(row))
        if direction == 0:
            direction = 1
        if pd.isna(atr) or atr <= 0 or entry <= 0:
            events.append(0)
            times.append(H)
            continue
        stop = entry - event_r * atr if direction > 0 else entry + event_r * atr
        event_hit = 0
        t_hit = H
        for j, (_, future) in enumerate(df15.iloc[i + 1:i + 1 + H].iterrows(), start=1):
            if treat_stop:
                if direction > 0 and _numeric(future.get("low")) <= stop:
                    event_hit = 1
                    t_hit = j
                    break
                if direction < 0 and _numeric(future.get("high")) >= stop:
                    event_hit = 1
                    t_hit = j
                    break
            if treat_choch and opposite_choch(dict(future), direction):
                event_hit = 1
                t_hit = j
                break
        events.append(event_hit)
        times.append(t_hit)

    return (
        pd.Series(events, index=df15.index, dtype=int, name="hazard_event"),
        pd.Series(times, index=df15.index, dtype=int, name="hazard_time"),
    )

