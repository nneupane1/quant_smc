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


def _bool_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float) != 0.0

    sval = s.astype(str).str.strip().str.lower()
    text_true = sval.isin({"1", "true", "yes", "up", "down", "long", "short"}).to_numpy(dtype=bool)
    num_true = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float) != 0.0
    return text_true | num_true


def _upper_text_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), "", dtype=object)
    arr = df[col].astype(str).str.strip().str.upper().to_numpy(dtype=object)
    invalid = (arr == "NAN") | (arr == "NONE") | (arr == "NULL")
    arr[invalid] = ""
    return arr


def _combined_bias(df: pd.DataFrame, *cols: str) -> np.ndarray:
    out = np.full(len(df), "", dtype=object)
    for col in cols:
        arr = _upper_text_series(df, col)
        mask = (out == "") & (arr != "")
        out[mask] = arr[mask]
    return out


def _infer_direction_array(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    direction = np.zeros(n, dtype=np.int8)

    side = _upper_text_series(df, "side")
    direction[side == "LONG"] = 1
    direction[side == "SHORT"] = -1

    unresolved = direction == 0
    if unresolved.any():
        flow = pd.to_numeric(
            df.get("flow_signal_1h", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        direction[unresolved & (flow > 0.0)] = 1
        direction[unresolved & (flow < 0.0)] = -1

    unresolved = direction == 0
    if unresolved.any():
        sweep_dir = pd.to_numeric(
            df.get("sweep_dir", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        direction[unresolved & (sweep_dir > 0.0)] = 1
        direction[unresolved & (sweep_dir < 0.0)] = -1

    unresolved = direction == 0
    if unresolved.any():
        sweep_low = _bool_series(df, "sweep_low")
        sweep_high = _bool_series(df, "sweep_high")
        direction[unresolved & sweep_low] = 1
        direction[unresolved & sweep_high] = -1

    unresolved = direction == 0
    if unresolved.any():
        bos_up = _bool_series(df, "bos_up")
        bos_down = _bool_series(df, "bos_down")
        direction[unresolved & bos_up] = 1
        direction[unresolved & bos_down] = -1

    unresolved = direction == 0
    if unresolved.any():
        bias = _combined_bias(df, "bias", "structure_bias", "structural_bias_6h")
        direction[unresolved & (bias == "UP")] = 1
        direction[unresolved & (bias == "DOWN")] = -1

    return direction


def _opposite_choch_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    choch_down = _bool_series(df, "choch_down")
    choch_up = _bool_series(df, "choch_up")
    choch_flag = _bool_series(df, "choch_flag")
    bias = _combined_bias(df, "bias", "structure_bias")
    long_opp = choch_down | (choch_flag & (bias == "DOWN"))
    short_opp = choch_up | (choch_flag & (bias == "UP"))
    return long_opp, short_opp


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
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # If present but fully degenerate, fall back to proxy.
            if float(s.abs().sum()) > 0.0:
                return s.clip(0.0, 1.0)
    return _confluence_proxy(df)


def hazard_series(df: pd.DataFrame) -> pd.Series:
    for col in ("hazard_score", "hazard"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if float(s.abs().sum()) > 0.0:
                return s.clip(0.0, 1.0)
    return _hazard_proxy(df)


def _first_numeric_series(df: pd.DataFrame, cols: Tuple[str, ...], default: float = 0.0) -> pd.Series:
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if s.notna().any():
                return s.fillna(default).astype(float)
    return pd.Series(float(default), index=df.index, dtype=float)


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x.clip(-20.0, 20.0)))


def _qscale_01(series: pd.Series, q_low: float = 0.10, q_high: float = 0.90) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 8:
        return pd.Series(0.5, index=s.index, dtype=float)
    lo = float(s.quantile(q_low))
    hi = float(s.quantile(q_high))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        med = float(s.median())
        mad = float((s - med).abs().median())
        if mad <= 1e-9 or not np.isfinite(mad):
            return pd.Series(0.5, index=s.index, dtype=float)
        z = (s.fillna(med) - med) / (1.4826 * mad)
        return _sigmoid(z)
    z = (s.fillna(lo) - lo) / max(hi - lo, 1e-9)
    z = (z - 0.5) * 4.0
    return _sigmoid(z)


def _confluence_proxy(df: pd.DataFrame) -> pd.Series:
    flow_strength = _first_numeric_series(df, ("flow_strength_1h",), 0.0).clip(0.0, 1.0)
    flow_ok = _first_numeric_series(df, ("flow_ok_1h",), 0.0).clip(0.0, 1.0)
    flow_sig = _first_numeric_series(df, ("flow_signal_1h",), 0.0).abs().clip(0.0, 1.0)
    flow = (0.60 * flow_strength + 0.25 * flow_ok + 0.15 * flow_sig).clip(0.0, 1.0)

    bos = _first_numeric_series(df, ("bos_flag", "bos_up", "bos_down"), 0.0).clip(0.0, 1.0)
    choch = _first_numeric_series(df, ("choch_flag", "choch_up", "choch_down"), 0.0).clip(0.0, 1.0)
    sweep = _first_numeric_series(df, ("sweep_flag", "liq_sweep"), 0.0).clip(0.0, 1.0)
    retest = _first_numeric_series(df, ("retest_fvg_ob_15m", "fresh_retest_15m"), 0.0).clip(0.0, 1.0)
    struct = (0.45 * bos + 0.25 * sweep + 0.15 * retest + 0.15 * (1.0 - choch)).clip(0.0, 1.0)

    trend = _first_numeric_series(df, ("p_regime_trend", "p_trend_up"), 0.0).clip(0.0, 1.0)
    expansion = _first_numeric_series(df, ("p_regime_expansion", "p_expansion"), 0.0).clip(0.0, 1.0)
    collapse = _first_numeric_series(df, ("p_regime_collapse",), 0.0).clip(0.0, 1.0)
    regime = (0.45 * trend + 0.35 * expansion + 0.20 * (1.0 - collapse)).clip(0.0, 1.0)

    absorb = _qscale_01(_first_numeric_series(df, ("absorption_score",), 0.0))
    session_mult = _first_numeric_series(df, ("session_quality_multiplier", "session_weight"), 1.0)
    session = ((session_mult - 0.75) / 0.40).clip(0.0, 1.0)

    conf_raw = (
        0.34 * flow
        + 0.22 * struct
        + 0.19 * regime
        + 0.15 * absorb
        + 0.10 * session
    ).clip(0.0, 1.0)
    conf = _qscale_01(conf_raw, q_low=0.20, q_high=0.98)
    return conf.astype(float)


def _hazard_proxy(df: pd.DataFrame) -> pd.Series:
    tox = _qscale_01(_first_numeric_series(df, ("toxicity_12h", "toxicity"), 0.0))
    vol = _qscale_01(_first_numeric_series(df, ("vol_zscore",), 0.0).abs())
    wick = _qscale_01(
        _first_numeric_series(df, ("wick_asymmetry_15m", "session_wick_asym_pct"), 0.0).abs()
    )
    sweep = _qscale_01(_first_numeric_series(df, ("liq_sweep_strength",), 0.0).abs())
    range_stress = _qscale_01(_first_numeric_series(df, ("range_pct",), 0.0))
    session_mult = _first_numeric_series(df, ("session_quality_multiplier", "session_weight"), 1.0)
    dead_zone = (session_mult < 0.90).astype(float)

    hazard = (
        0.34 * tox
        + 0.24 * vol
        + 0.16 * wick
        + 0.14 * sweep
        + 0.07 * dead_zone
        + 0.05 * range_stress
    ).clip(0.0, 1.0)
    hazard = (0.75 * hazard).clip(0.0, 1.0)
    return hazard.astype(float)


def _evr_series(df: pd.DataFrame, conf: pd.Series, hazard: pd.Series) -> pd.Series:
    if "evr" in df.columns:
        evr = pd.to_numeric(df["evr"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if float(evr.abs().sum()) > 0.0:
            return evr

    flow = _first_numeric_series(df, ("flow_strength_1h",), 0.0).clip(0.0, 1.0)
    trend = _first_numeric_series(df, ("p_regime_trend",), 0.0).clip(0.0, 1.0)
    disp = _qscale_01(_first_numeric_series(df, ("displacement_body_pct_1h", "displacement_15m"), 0.0))
    demand = _first_numeric_series(df, ("demand_quality_1h", "demand_quality"), 0.5).clip(0.0, 1.0)
    supply = _first_numeric_series(df, ("supply_quality_1h", "supply_quality"), 0.5).clip(0.0, 1.0)
    zone = (0.5 * demand + 0.5 * supply).clip(0.0, 1.0)
    session_mult = _first_numeric_series(df, ("session_quality_multiplier", "session_weight"), 1.0)
    session = ((session_mult - 0.75) / 0.40).clip(0.0, 1.0)

    evr = (
        -0.60
        + 2.40 * conf
        + 1.40 * flow
        + 0.80 * trend
        + 0.60 * disp
        + 0.50 * session
        + 0.40 * zone
        - 2.10 * hazard
    ).clip(-2.0, 6.0)
    return evr.astype(float)


def _median_r_series(df: pd.DataFrame, conf: pd.Series, hazard: pd.Series) -> pd.Series:
    for col in ("median_r", "median_R"):
        if col in df.columns:
            med = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if float(med.abs().sum()) > 0.0:
                return med

    flow = _first_numeric_series(df, ("flow_strength_1h",), 0.0).clip(0.0, 1.0)
    disp = _qscale_01(_first_numeric_series(df, ("displacement_body_pct_1h", "displacement_15m"), 0.0))
    demand = _first_numeric_series(df, ("demand_quality_1h", "demand_quality"), 0.5).clip(0.0, 1.0)
    supply = _first_numeric_series(df, ("supply_quality_1h", "supply_quality"), 0.5).clip(0.0, 1.0)
    zone = (0.5 * demand + 0.5 * supply).clip(0.0, 1.0)

    medr = (
        1.0
        + 6.5 * conf
        + 2.0 * flow
        + 1.0 * disp
        + 0.8 * zone
        - 4.0 * hazard
    ).clip(0.0, 12.0)
    return medr.astype(float)


def _tier_series(
    df: pd.DataFrame,
    conf: pd.Series,
    evr: pd.Series,
    medr: pd.Series,
    hazard: pd.Series,
    conf_min: float,
    evr_min: float,
    medr_min: float,
    hazard_cap: float,
) -> pd.Series:
    if "tier" in df.columns:
        tier = df["tier"].astype(str).str.upper()
        # If this field already carries explicit execution tiers, keep it.
        if tier.isin({"A+", "A", "B", "SKIP"}).any():
            return tier.fillna("")

    aplus = (
        (conf >= conf_min)
        & (evr >= evr_min)
        & (medr >= medr_min)
        & (hazard <= hazard_cap)
    )
    a = (
        (conf >= max(conf_min - 0.18, 0.58))
        & (evr >= max(1.0, evr_min * 0.70))
        & (medr >= max(3.0, medr_min * 0.70))
        & (hazard <= min(0.50, hazard_cap + 0.10))
    )
    tier = np.where(aplus, "A+", np.where(a, "A", "B"))
    return pd.Series(tier, index=df.index, dtype="object")


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
    evr = _evr_series(df15, conf, hazard)
    medr = _median_r_series(df15, conf, hazard)
    tier = _tier_series(
        df15,
        conf=conf,
        evr=evr,
        medr=medr,
        hazard=hazard,
        conf_min=conf_min,
        evr_min=evr_min,
        medr_min=medr_min,
        hazard_cap=haz_cap,
    ).astype(str).str.upper()

    candidate = (
        ((conf >= conf_min) | tier.eq("A+"))
        & (evr >= evr_min)
        & (medr >= medr_min)
        & (hazard <= haz_cap)
    ).astype(int)
    # label[i] = any(candidate[i+1 : i+1+H]); compute with reverse rolling max.
    labels = (
        candidate.iloc[::-1]
        .shift(1)
        .rolling(H, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(int)
        .iloc[::-1]
    )
    labels.name = "label_eop"
    return labels


def compute_edp_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    atr_col = resolve_atr_col(df15)
    if atr_col is None:
        return pd.Series(0, index=df15.index, dtype=int, name="label_edp")

    H = max(int(cfg.get("horizon_bars", 96)), 1)
    dd_r = abs(float(cfg.get("drawdown_R_threshold", -3.0)))
    n = len(df15)
    if n == 0:
        return pd.Series([], index=df15.index, dtype=int, name="label_edp")

    atr = pd.to_numeric(df15.get(atr_col, np.nan), errors="coerce").to_numpy(dtype=float)
    entry = pd.to_numeric(df15.get("close", np.nan), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df15.get("low", np.nan), errors="coerce").to_numpy(dtype=float)
    valid = (~np.isnan(atr)) & (atr > 0.0) & (~np.isnan(entry)) & (entry > 0.0)
    target = entry - (dd_r * atr)

    hit = np.zeros(n, dtype=bool)
    for j in range(1, H + 1):
        upto = n - j
        if upto <= 0:
            break
        cond = low[j:] <= target[:upto]
        idx = np.nonzero((~hit[:upto]) & cond & valid[:upto])[0]
        if idx.size:
            hit[idx] = True

    labels = (hit & valid).astype(int)
    return pd.Series(labels, index=df15.index, dtype=int, name="label_edp")


def compute_hazard_labels(df15: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    atr_col = resolve_atr_col(df15)
    H = max(int(cfg.get("horizon_bars", 48)), 1)
    if atr_col is None:
        return (
            pd.Series(0, index=df15.index, dtype=int, name="hazard_event"),
            pd.Series(H, index=df15.index, dtype=int, name="hazard_time"),
        )

    event_r = abs(float(cfg.get("event_R_threshold", -1.0)))
    treat_choch = bool(cfg.get("treat_choch_as_event", True))
    treat_stop = bool(cfg.get("treat_stop_as_event", True))

    n = len(df15)
    if n == 0:
        return (
            pd.Series([], index=df15.index, dtype=int, name="hazard_event"),
            pd.Series([], index=df15.index, dtype=int, name="hazard_time"),
        )

    atr = pd.to_numeric(df15.get(atr_col, np.nan), errors="coerce").to_numpy(dtype=float)
    entry = pd.to_numeric(df15.get("close", np.nan), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df15.get("low", np.nan), errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df15.get("high", np.nan), errors="coerce").to_numpy(dtype=float)

    direction = _infer_direction_array(df15).astype(int)
    direction[direction == 0] = 1
    valid = (~np.isnan(atr)) & (atr > 0.0) & (~np.isnan(entry)) & (entry > 0.0)
    stop = np.where(direction > 0, entry - (event_r * atr), entry + (event_r * atr))

    opp_long, opp_short = _opposite_choch_arrays(df15)
    event = np.zeros(n, dtype=bool)
    t_hit = np.full(n, H, dtype=int)

    for j in range(1, H + 1):
        upto = n - j
        if upto <= 0:
            break
        dir_i = direction[:upto]
        cond = np.zeros(upto, dtype=bool)

        if treat_stop:
            cond_stop_long = low[j:] <= stop[:upto]
            cond_stop_short = high[j:] >= stop[:upto]
            cond |= np.where(dir_i > 0, cond_stop_long, cond_stop_short)

        if treat_choch:
            cond_choch_long = opp_long[j:]
            cond_choch_short = opp_short[j:]
            cond |= np.where(dir_i > 0, cond_choch_long, cond_choch_short)

        idx = np.nonzero((~event[:upto]) & cond & valid[:upto])[0]
        if idx.size:
            event[idx] = True
            t_hit[idx] = j

    event &= valid
    t_hit[~valid] = H

    return (
        pd.Series(event.astype(int), index=df15.index, dtype=int, name="hazard_event"),
        pd.Series(t_hit.astype(int), index=df15.index, dtype=int, name="hazard_time"),
    )
