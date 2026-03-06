"""
feature_builder.py — full multi-asset, multi-TF feature assembly (leak-safe).

Builds unified 15m-level feature frame:
 - loads asset-specific TF data
 - attaches SMC signals (swings, bos/choch, zones, fvg, sweep)
 - EMA features (multi-TF)
 - liquidity features
 - volatility features
 - regime features (6h/12h HMM/HDBSCAN outputs)
 - rolling-window z-scores
 - session weight
Also emits an HTF SMC audit table (unpruned, unlagged) for dashboards.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import argparse
import time
from pathlib import Path
import json

from quant_system.config.config_loader import ConfigLoader
from quant_system.features.smc.swings import SwingHighLowDetector
from quant_system.features.smc.bos_choch import BOSCHOCHDetector
from quant_system.features.smc.fvg import FVGDetector
from quant_system.features.smc.sweep import LiquiditySweepDetector
from quant_system.features.smc.zones import OrderBlockDetector
from quant_system.features.smc.structure_context import StructureContextBuilder

from quant_system.features.ema_features import EMAFeatureBuilder as EMAFeatureBlock
from quant_system.features.liquidity_features import LiquidityFeatureBuilder as LiquidityFeatureBlock
from quant_system.features.volatility_features import VolatilityFeatureBuilder as VolatilityFeatureBlock
from quant_system.features.regime_features import RegimeFeatureBlock
from quant_system.features.rolling_windows import RollingWindows
from quant_system.features.absorption_features import (
    AbsorptionFeatureBuilder,
    AbsorptionConfig,
)

from quant_system.utils.time_utils import SessionClassifier as SessionTimeFeature  # placeholder
from quant_system.utils.logger import get_logger

LOG = get_logger("feature_builder")


class FeatureBuilder:
    """
    Builds complete multi-TF feature matrix at 15m resolution.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.cfg = config_loader
        self.features_cfg = self.cfg.load_yaml("features.yaml").get("features", {})
        self.asset_cfg = self.cfg.load_yaml("assets.yaml")
        self.audit_table: Optional[pd.DataFrame] = None

        # SMC components
        self.swings_15 = SwingHighLowDetector(left=2, right=2)
        self.swings_1h = SwingHighLowDetector(left=3, right=3)
        self.swings_6h = SwingHighLowDetector(left=3, right=3)
        self.swings_12h = SwingHighLowDetector(left=4, right=4)

        self.bos = BOSCHOCHDetector()
        self.fvg = FVGDetector()
        self.sweep = LiquiditySweepDetector()
        self.zones = OrderBlockDetector()
        self.context = StructureContextBuilder(self.cfg.load_yaml("features.yaml"))

        # EMA / Liquidity / Vol / Regime
        self.ema_block = EMAFeatureBlock(config_loader)
        self.liq_block = LiquidityFeatureBlock(config_loader)
        self.vol_block = VolatilityFeatureBlock()
        self.reg_block = RegimeFeatureBlock(config_loader)
        # Absorption (iceberg proxy) block
        absorb_cfg_raw = self.features_cfg.get("absorption", {}) if isinstance(self.features_cfg, dict) else {}
        absorb_cfg = AbsorptionConfig(
            enable=absorb_cfg_raw.get("enable", True),
            window_minutes=absorb_cfg_raw.get("window_minutes", 60),
            use_spread_ofi=absorb_cfg_raw.get("use_spread_ofi", True),
            veto_threshold=absorb_cfg_raw.get("veto_threshold", 0.70),
            throttle_band=tuple(absorb_cfg_raw.get("throttle_band", (0.40, 0.70))),
        )
        self.absorb_block = AbsorptionFeatureBuilder(absorb_cfg)

        self.session_block = SessionTimeFeature()

        # Optional lag/rolling config
        self.lag_cfg = self.features_cfg.get("lagging", {})

        LOG.info("[FeatureBuilder] Initialized successfully")

    @staticmethod
    def _normalize_df(df: pd.DataFrame, path_hint: str = "") -> pd.DataFrame:
        """
        Normalize column names to lowercase and ensure both:
          - 'dt' datetime column (UTC)
          - 'timestamp' numeric seconds column
        Accepts timestamp/ts/dt/time columns (string or numeric epoch).
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        time_col = None
        for cand in ("dt", "timestamp", "ts", "time"):
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            raise ValueError(f"No time column found in dataframe {path_hint or ''}")
        series = df[time_col]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(numeric) else 0.0

        if numeric_ratio >= 0.95:
            # Interpret numeric epochs by magnitude to avoid the default ns parser trap.
            # Typical ranges:
            #   seconds ~ 1e9
            #   milliseconds ~ 1e12
            #   microseconds ~ 1e15
            #   nanoseconds ~ 1e18
            abs_med = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
            if abs_med >= 1e17:
                unit = "ns"
            elif abs_med >= 1e14:
                unit = "us"
            elif abs_med >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(series, utc=True, errors="coerce")

        if dt.isna().all():
            raise ValueError(f"Failed to parse any timestamps from column '{time_col}' in {path_hint or 'dataframe'}")

        df["dt"] = dt
        # integer seconds since epoch for swing detector
        df["timestamp"] = (dt.astype("int64") // 10**9).astype("int64")
        return df

    def _paths(self, out_path: Path, asset: str) -> Dict[str, Path]:
        base = out_path.parent
        base.mkdir(parents=True, exist_ok=True)
        return {
            "checkpoint": base / f"{asset}_feat_checkpoint.json",
            "stage1": base / f"{asset}_stage1_smc.csv",
            "stage2": base / f"{asset}_stage2_htf.csv",
            "stage3": base / f"{asset}_stage3_core.csv",
            "audit": base / f"{asset}_htf_audit.csv",
        }

    def _load_checkpoint(self, ckpt_path: Path) -> Dict[str, any]:
        if ckpt_path.exists():
            try:
                return json.loads(ckpt_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_checkpoint(self, ckpt_path: Path, asset: str, input_dir: str, out_path: Path, stage: int):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps({
            "asset": asset,
            "input_dir": str(input_dir),
            "out_path": str(out_path),
            "last_stage": stage,
        }, indent=2))

    def _load_from_dir(self, input_dir: str, asset: str) -> Dict[str, pd.DataFrame]:
        base = Path(input_dir)
        dfs: Dict[str, pd.DataFrame] = {}
        for tf in ["15m", "1h", "6h", "12h"]:
            path = base / f"{asset}_{tf}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing timeframe file: {path}")
            dfs[tf] = self._normalize_df(pd.read_csv(path), path_hint=str(path))
        return dfs

    def build_from_dir(self, input_dir: str, asset: str, out_path: Optional[str] = None) -> pd.DataFrame:
        paths: Dict[str, Path] = {}
        ckpt = {}
        if out_path:
            paths = self._paths(Path(out_path), asset)
            ckpt = self._load_checkpoint(paths["checkpoint"])

        dfs = self._load_from_dir(input_dir, asset)
        df15 = dfs["15m"].copy()
        df1h = dfs["1h"]
        df6h = dfs["6h"]
        df12 = dfs["12h"]

        # Stage 1: 15m SMC + helper flags
        if ckpt.get("last_stage", 0) >= 1 and paths.get("stage1") and paths["stage1"].exists():
            df15 = pd.read_csv(paths["stage1"], parse_dates=["dt"])
            LOG.info(f"[FeatureBuilder] Stage 1 cache hit -> {paths['stage1']}")
        else:
            t0 = time.perf_counter()
            LOG.info("[FeatureBuilder] Stage 1: 15m SMC (swings/BOS/FVG/sweep/zones)")
            df15 = self._normalize_df(df15, "15m")
            df15 = self.swings_15.apply(df15)
            df15 = self.bos.apply(df15)
            df15 = self.fvg.apply(df15)
            df15 = self.sweep.apply(df15)
            df15 = self.zones.apply(df15)
            df15 = self._add_helper_flags(df15)
            if paths.get("stage1"):
                df15.to_csv(paths["stage1"], index=False)
                self._save_checkpoint(paths["checkpoint"], asset, input_dir, Path(out_path) if out_path else Path(), 1)
            LOG.info(f"[FeatureBuilder] Stage 1 done in {time.perf_counter() - t0:.2f}s")

        # Normalize HTFs (always needed downstream)
        df1h = self._normalize_df(df1h, "1h")
        df6h = self._normalize_df(df6h, "6h")
        df12 = self._normalize_df(df12, "12h")

        # Stage 2: HTF SMC + joins + audit
        if ckpt.get("last_stage", 0) >= 2 and paths.get("stage2") and paths["stage2"].exists():
            df15 = pd.read_csv(paths["stage2"], parse_dates=["dt"])
            if paths.get("audit") and paths["audit"].exists():
                self.audit_table = pd.read_csv(paths["audit"], parse_dates=["dt"])
            if "displacement_body_pct_1h" in df15.columns and "flow_signal_1h" in df15.columns:
                LOG.info(f"[FeatureBuilder] Stage 2 cache hit -> {paths['stage2']}")
            else:
                LOG.info("[FeatureBuilder] Stage 2 cache stale for 1h flow features; rebuilding.")
                ckpt["last_stage"] = 1
        if ckpt.get("last_stage", 0) < 2:
            t0 = time.perf_counter()
            LOG.info("[FeatureBuilder] Stage 2: HTF SMC + joins")
            htf_full = {
                "1h": self._apply_smc_full(df1h, "1h"),
                "6h": self._apply_smc_full(df6h, "6h"),
                "12h": self._apply_smc_full(df12, "12h"),
            }
            self.audit_table = self._build_audit_table(htf_full, asset)
            df1h_s = self._apply_smc_pruned(htf_full["1h"])
            df6h_s = self._apply_smc_pruned(htf_full["6h"])
            df12_s = self._apply_smc_pruned(htf_full["12h"])
            df15 = self.context.apply(df15, htf_full["6h"])
            df15 = self._join_tf(df15, df1h_s, "1h")
            df15 = self._join_tf(df15, self._build_flow_1h(df1h), "1h")
            df15 = self._join_tf(df15, df6h_s, "6h")
            df15 = self._join_tf(df15, df12_s, "12h")
            if paths.get("stage2"):
                df15.to_csv(paths["stage2"], index=False)
                if self.audit_table is not None and not self.audit_table.empty:
                    self.audit_table.to_csv(paths["audit"], index=False)
                self._save_checkpoint(paths["checkpoint"], asset, input_dir, Path(out_path) if out_path else Path(), 2)
            LOG.info(f"[FeatureBuilder] Stage 2 done in {time.perf_counter() - t0:.2f}s")

        # Stage 3: EMA/Liq/Vol/Absorption/Regime
        if ckpt.get("last_stage", 0) >= 3 and paths.get("stage3") and paths["stage3"].exists():
            df15 = pd.read_csv(paths["stage3"], parse_dates=["dt"])
            if "flow_strength_1h" in df15.columns and "p_regime_trend" in df15.columns:
                LOG.info(f"[FeatureBuilder] Stage 3 cache hit -> {paths['stage3']}")
            else:
                LOG.info("[FeatureBuilder] Stage 3 cache stale; rebuilding.")
                ckpt["last_stage"] = 2
        if ckpt.get("last_stage", 0) < 3:
            t0 = time.perf_counter()
            LOG.info("[FeatureBuilder] Stage 3: EMA/Liq/Vol/Absorption/Regime")
            df15 = self.ema_block.apply(df15, df1h, df6h, df12)
            df15 = self.liq_block.apply(df15)
            df15 = self.vol_block.apply(df15)
            df15 = self.absorb_block.apply(df15)
            df15 = self.reg_block.apply(df15, df6h, df12, asset)
            if paths.get("stage3"):
                df15.to_csv(paths["stage3"], index=False)
                self._save_checkpoint(paths["checkpoint"], asset, input_dir, Path(out_path) if out_path else Path(), 3)
            LOG.info(f"[FeatureBuilder] Stage 3 done in {time.perf_counter() - t0:.2f}s")

        # Stage 4: Session, lagging, rolling, cleanup
        LOG.info("[FeatureBuilder] Stage 4: session/lag/rolling/cleanup")
        t0 = time.perf_counter()
        df15 = self.session_block.apply(df15)
        df15 = self._ensure_session_weight(df15)
        df15 = self._lag_event_flags(df15)
        df15 = self._add_lagged_features(df15)
        core_subset = [c for c in ["dt", "timestamp", "open", "high", "low", "close"] if c in df15.columns]
        df15 = df15.dropna(subset=core_subset).reset_index(drop=True)

        if paths.get("checkpoint"):
            self._save_checkpoint(paths["checkpoint"], asset, input_dir, Path(out_path) if out_path else Path(), 4)
        LOG.info(f"[FeatureBuilder] Stage 4 done in {time.perf_counter() - t0:.2f}s")

        LOG.info(f"[FeatureBuilder] Feature build complete rows={len(df15)} for asset={asset}")
        return df15

    # ----------------------------------------------------------------------
    # BUILD FEATURES FOR ONE ASSET (FULL PIPELINE)
    # ----------------------------------------------------------------------
    def build(
        self,
        dfs: Dict[str, pd.DataFrame],
        asset: str
    ) -> pd.DataFrame:
        """
        dfs = {
            "1m":  df1m,
            "15m": df15m,
            "1h":  df1h,
            "6h":  df6h,
            "12h": df12h
        }
        """
        LOG.info(f"[FeatureBuilder] Building features for asset={asset}")

        df15 = dfs["15m"].copy()
        df1h = dfs["1h"]
        df6h = dfs["6h"]
        df12 = dfs["12h"]

        # Normalize columns once here for downstream SMC expectations
        df15 = self._normalize_df(df15, "15m")
        df1h = self._normalize_df(df1h, "1h")
        df6h = self._normalize_df(df6h, "6h")
        df12 = self._normalize_df(df12, "12h")

        LOG.info("[FeatureBuilder] Attaching SMC signals (15m)")
        df15 = self.swings_15.apply(df15)
        df15 = self.bos.apply(df15)
        df15 = self.fvg.apply(df15)
        df15 = self.sweep.apply(df15)
        df15 = self.zones.apply(df15)
        df15 = self._add_helper_flags(df15)

        LOG.info("[FeatureBuilder] Processing higher-TF SMC (1h/6h/12h)")
        htf_full = {
            "1h": self._apply_smc_full(df1h, "1h"),
            "6h": self._apply_smc_full(df6h, "6h"),
            "12h": self._apply_smc_full(df12, "12h"),
        }

        # Build audit table (unpruned, unlagged)
        self.audit_table = self._build_audit_table(htf_full, asset)

        # Prune HTF signals for modeling (lean set)
        df1h_s = self._apply_smc_pruned(htf_full["1h"])
        df6h_s = self._apply_smc_pruned(htf_full["6h"])
        df12_s = self._apply_smc_pruned(htf_full["12h"])

        # Structural context from processed 6h frame
        df15 = self.context.apply(df15, htf_full["6h"])

        df15 = self._join_tf(df15, df1h_s, "1h")
        df15 = self._join_tf(df15, self._build_flow_1h(df1h), "1h")
        df15 = self._join_tf(df15, df6h_s, "6h")
        df15 = self._join_tf(df15, df12_s, "12h")

        # EMA features
        LOG.info("[FeatureBuilder] Attaching EMA block")
        df15 = self.ema_block.apply(df15, df1h, df6h, df12)

        # Liquidity features
        LOG.info("[FeatureBuilder] Attaching liquidity block")
        df15 = self.liq_block.apply(df15)

        # Volatility block
        LOG.info("[FeatureBuilder] Attaching volatility block")
        df15 = self.vol_block.apply(df15)

        # Absorption (iceberg proxy) block
        LOG.info("[FeatureBuilder] Attaching absorption (iceberg) block")
        df15 = self.absorb_block.apply(df15)

        # Regime block (HMM/HDBSCAN external)
        LOG.info("[FeatureBuilder] Attaching regime block")
        df15 = self.reg_block.apply(df15, df6h, df12, asset)

        # Session window
        LOG.info("[FeatureBuilder] Attaching session features")
        df15 = self.session_block.apply(df15)
        df15 = self._ensure_session_weight(df15)

        # Leak guard on local events
        LOG.info("[FeatureBuilder] Shifting event flags by +1 bar to avoid look-ahead")
        df15 = self._lag_event_flags(df15)

        # Rolling / lagged features (time-series safe)
        LOG.info("[FeatureBuilder] Adding lag/rolling windows")
        df15 = self._add_lagged_features(df15)

        # Final cleanup: only drop rows missing core price/time fields to avoid nuking the dataset
        core_subset = [c for c in ["dt", "timestamp", "open", "high", "low", "close"] if c in df15.columns]
        df15 = df15.dropna(subset=core_subset).reset_index(drop=True)

        LOG.info(f"[FeatureBuilder] Feature build complete rows={len(df15)} for asset={asset}")
        return df15

    # ----------------------------------------------------------------------
    # JOIN HIGHER TF BAR VALUES INTO 15M ROWS
    # ----------------------------------------------------------------------
    def _join_tf(self, df15: pd.DataFrame, df_tf: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Creates suffix columns such as:
          swing_high_1h, swing_low_6h, bos_flag_12h, ...
        Only uses closed bars (right-open).
        """
        df_tf = df_tf.rename(columns=lambda c: f"{c}_{label}" if c not in ["dt"] else "dt")
        merged = pd.merge_asof(
            df15.sort_values("dt"),
            df_tf.sort_values("dt"),
            on="dt",
            direction="backward",
            allow_exact_matches=False,
        )
        return merged

    def _apply_smc_full(self, df_tf: pd.DataFrame, tf_label: str) -> pd.DataFrame:
        """Apply full SMC on a TF (no pruning/lagging)."""
        x = df_tf.copy()
        if tf_label == "1h":
            x = self.swings_1h.apply(x)
        elif tf_label == "6h":
            x = self.swings_6h.apply(x)
        else:  # "12h"
            x = self.swings_12h.apply(x)
        x = self.bos.apply(x)
        x = self.fvg.apply(x)
        x = self.sweep.apply(x)
        x = self.zones.apply(x)
        return x

    def _apply_smc_pruned(self, df_tf: pd.DataFrame) -> pd.DataFrame:
        """
        Keep lean HTF fields for modeling to avoid bloat.
        """
        keep: List[str] = ["dt"]
        for cand in [
            "bos_flag",
            "choch_flag",
            "bias",
            "structure_bias",
            "sweep_flag",
            "fvg_mid",
            "fvg_mid_price",
            "fvg_hi",
            "fvg_lo",
            "zone_id",
            "zone_hi",
            "zone_lo",
            "zone_high",
            "zone_low",
            "demand_quality",
            "supply_quality",
            "demand_age",
            "supply_age",
        ]:
            if cand in df_tf.columns:
                keep.append(cand)
        for alt in ["ob_id", "structural_bias_6h", "zone_recency", "zone_displacement", "zone_mitigation", "zone_pd", "zone_ema_align"]:
            if alt in df_tf.columns and alt not in keep:
                keep.append(alt)
        return df_tf[keep].copy()

    def _build_flow_1h(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Derive 1h flow/impulse features from raw Kraken bars.
        These are projected onto the 15m execution frame and used by both
        gating and the dedicated 1h flow model.
        """
        if df_1h is None or df_1h.empty:
            return pd.DataFrame(columns=["dt"])

        out = df_1h[["dt"]].copy()
        rng = (df_1h["high"] - df_1h["low"]).replace(0, np.nan)
        body = (df_1h["close"] - df_1h["open"]).abs()
        signed_body = df_1h["close"] - df_1h["open"]
        ret = df_1h["close"].pct_change().fillna(0.0)
        volume = df_1h["volume"].astype(float)
        volume_mean = volume.rolling(48, min_periods=12).mean()
        volume_std = volume.rolling(48, min_periods=12).std().replace(0, np.nan)
        volume_z = ((volume - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_pct = ((df_1h["high"] - df_1h["low"]) / df_1h["close"].replace(0, np.nan)).fillna(0.0)
        range_mean = range_pct.rolling(48, min_periods=12).mean()
        range_std = range_pct.rolling(48, min_periods=12).std().replace(0, np.nan)
        range_z = ((range_pct - range_mean) / range_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        close_loc = ((df_1h["close"] - df_1h["low"]) / rng).clip(0.0, 1.0).fillna(0.5)

        disp_body_pct = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        direction = np.sign(signed_body).astype(int)

        impulse_up = (
            (direction > 0)
            & (disp_body_pct >= 0.60)
            & (volume_z >= 0.80)
            & (close_loc >= 0.65)
        )
        impulse_down = (
            (direction < 0)
            & (disp_body_pct >= 0.60)
            & (volume_z >= 0.80)
            & (close_loc <= 0.35)
        )

        flow_signal = pd.Series(0, index=df_1h.index, dtype=int)
        flow_signal = flow_signal.mask(impulse_up, 1)
        flow_signal = flow_signal.mask(impulse_down, -1)

        freshness = []
        last_impulse_idx = None
        for i, sig in enumerate(flow_signal.tolist()):
            if sig != 0:
                last_impulse_idx = i
                freshness.append(0)
            elif last_impulse_idx is None:
                freshness.append(999)
            else:
                freshness.append(i - last_impulse_idx)

        out["displacement_body_pct"] = disp_body_pct.fillna(0.0)
        out["body_dir"] = direction
        out["ret_1h"] = ret
        out["close_loc"] = close_loc
        out["range_pct"] = range_pct
        out["range_z"] = range_z
        out["volume_z"] = volume_z
        out["flow_signal"] = flow_signal
        out["flow_age_bars"] = freshness
        out["flow_ok"] = ((flow_signal != 0) | (pd.Series(freshness, index=df_1h.index) <= 4)).astype(int)
        out["flow_strength"] = (
            0.55 * out["displacement_body_pct"].clip(0, 1)
            + 0.25 * out["volume_z"].clip(-1, 3).div(3.0)
            + 0.20 * out["range_z"].clip(-1, 3).div(3.0)
        ).clip(0.0, 1.0)
        return out

    def _build_audit_table(self, htf: Dict[str, pd.DataFrame], asset: str) -> pd.DataFrame:
        """
        Create an audit-friendly HTF SMC table (unpruned, unlagged) for dashboards.
        """
        frames = []
        keep_candidates = [
            "dt", "bos_flag", "choch_flag", "bias", "sweep_flag",
            "fvg_mid", "fvg_hi", "fvg_lo", "fvg_mid_price",
            "zone_id", "zone_high", "zone_low", "zone_hi", "zone_lo",
        ]
        for tf, df_tf in htf.items():
            cols = ["dt"] + [c for c in keep_candidates if c in df_tf.columns and c != "dt"]
            tmp = df_tf[cols].copy()
            tmp["tf"] = tf
            frames.append(tmp)
        if not frames:
            return pd.DataFrame()
        audit = pd.concat(frames, ignore_index=True).sort_values(["dt", "tf"])
        audit["asset"] = asset
        return audit.reset_index(drop=True)

    def _add_helper_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lightweight helper flags used by pyramiding and adds:
          - displacement_15m: big body vs range
          - fresh_retest_15m: quick proxy for a fresh touch of OB/FVG/sweep
          - retest_fvg_ob_15m: alias for same concept
        """
        if df is None or df.empty:
            return df
        out = df.copy()
        body = (out["close"] - out["open"]).abs()
        rng = (out["high"] - out["low"]).replace(0, 1e-9)
        body_pct = body / rng
        out["displacement_15m"] = (body_pct >= 0.55).astype(int)

        retouch_cols = [c for c in ["fvg_touch_flag", "fvg_filled_flag", "sweep_flag"] if c in out.columns]
        if retouch_cols:
            retest = out[retouch_cols].max(axis=1)
        else:
            retest = pd.Series(0, index=out.index)
        out["fresh_retest_15m"] = (retest > 0).astype(int)
        out["retest_fvg_ob_15m"] = out["fresh_retest_15m"]
        return out

    def _ensure_session_weight(self, df15: pd.DataFrame) -> pd.DataFrame:
        """
        Create 'session_weight' ∈ {1.0 (LDN/NY), 0.2 (off-hours)}.
        """
        weight = pd.Series(0.2, index=df15.index, dtype=float)
        if "is_ldn" in df15.columns:
            weight = weight.mask(df15["is_ldn"] == 1, 1.0)
        if "is_ny" in df15.columns:
            weight = weight.mask(df15["is_ny"] == 1, 1.0)
        if "session_london" in df15.columns:
            weight = weight.mask(df15["session_london"] == 1, 1.0)
        if "session_ny" in df15.columns:
            weight = weight.mask(df15["session_ny"] == 1, 1.0)
        if "session_overlap" in df15.columns:
            weight = weight.mask(df15["session_overlap"] == 1, 1.10)
        for col in ["session", "session_name"]:
            if col in df15.columns:
                weight = weight.mask(df15[col].isin(["LDN", "NY", "london", "newyork", "NewYork"]), 1.0)
        df15["session_weight"] = weight
        return df15

    def _lag_event_flags(self, df15: pd.DataFrame) -> pd.DataFrame:
        """
        Shift local 15m event flags by +1 bar to avoid look-ahead.
        """
        event_cols = [
            "bos_flag",
            "choch_flag",
            "sweep_flag",
            "fvg_touch",
            "fvg_touch_flag",
            "fvg_filled_flag",
        ]
        present = [c for c in event_cols if c in df15.columns]
        if not present:
            return df15
        out = df15.copy()
        for c in present:
            shifted = out[c].shift(1)
            if set(out[c].dropna().unique()).issubset({0, 1}):
                shifted = shifted.fillna(0).astype(int)
            out[f"{c}_lag1"] = shifted
        return out

    # ----------------------------------------------------------------------
    # Lag / rolling block
    # ----------------------------------------------------------------------
    def _add_lagged_features(self, df15: pd.DataFrame) -> pd.DataFrame:
        """
        Adds configurable lags and rolling stats to selected columns to keep the
        time-series ordering intact and avoid leakage.
        """
        if df15 is None or df15.empty:
            return df15

        cfg = self.lag_cfg
        lags = cfg.get("lags", [1, 2, 3, 4])
        roll_windows = cfg.get("rolling_windows", [4, 8, 16])
        cols = cfg.get("columns")

        if not cols:
            default_cols = [
                "close",
                "high",
                "low",
                "open",
                "volume",
                "atr",
            ]
            cols = [c for c in default_cols if c in df15.columns]
        else:
            cols = [c for c in cols if c in df15.columns]

        if not cols:
            return df15

        out = RollingWindows.add_lags(df15, cols, lags) if lags else df15
        out = RollingWindows.add_rolling_stats(out, cols, roll_windows) if roll_windows else out
        return out


def _cli():
    parser = argparse.ArgumentParser(description="Build features from TF CSVs.")
    parser.add_argument("--config", default="quant_system/config", help="Config directory")
    parser.add_argument("--input", required=True, help="Input directory containing {ASSET}_{15m,1h,6h,12h}.csv")
    parser.add_argument("--asset", default="XBTUSD", help="Asset symbol prefix used in filenames")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    fb = FeatureBuilder(cfg)
    out_path = Path(args.out)
    feat = fb.build_from_dir(args.input, args.asset, out_path=str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False)
    LOG.info(f"[FeatureBuilder] Wrote features to {out_path}")

    # Optional HTF audit export
    if fb.audit_table is not None and not fb.audit_table.empty:
        audit_path = out_path.parent / f"{args.asset}_htf_audit.csv"
        fb.audit_table.to_csv(audit_path, index=False)
        LOG.info(f"[FeatureBuilder] Wrote HTF SMC audit to {audit_path}")


if __name__ == "__main__":
    _cli()
