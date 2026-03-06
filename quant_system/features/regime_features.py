"""
Regime feature engineering:
    - volatility-normalized metrics
    - trend persistence
    - compression / expansion signals
    - aggregated structural features (6h / 12h)
    - regime-proxy inputs for HDBSCAN + HMM
    - RSV precursor features

These features feed the:
    - unsupervised clustering (HDBSCAN)
    - HMM smoothing
    - supervised auxiliary regime classifier
    - confluence engine (RSV)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import hdbscan
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None  # type: ignore

from quant_system.utils.logger import get_logger, log
from quant_system.features.rolling_windows import RollingWindows

LOG = get_logger("regime_block")


class RegimeFeatures:
    """
    Builds multi-timeframe regime input block.
    Works on 12h, 6h, 1h, then projected to 15m rows.
    """

    def __init__(self, config: Dict[str, Any]):
        rcfg = config["features"]["regime"]

        self.cluster_lb = int(rcfg["cluster_features_lookback"])
        self.vol_pct_lb = int(rcfg["volatility_percentile_window"])
        self.tox_lb = int(rcfg["toxicity_window"])
        self.comp_lb = int(rcfg["compression_lookback"])

        log("RegimeFeatures initialized.")

    # ------------------------------------------------------------
    # Compute volatility percentile (rolling)
    # ------------------------------------------------------------
    def _vol_percentile(self, df: pd.DataFrame) -> pd.Series:
        vol = df["vol_zscore"].abs()
        pct = vol.rolling(self.vol_pct_lb).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan,
            raw=False
        )
        return pct

    # ------------------------------------------------------------
    # Trend persistence
    # ------------------------------------------------------------
    def _trend_persistence(self, df: pd.DataFrame) -> pd.Series:
        ret = df["close"].pct_change()
        up = (ret > 0).astype(int)
        ratio = up.rolling(self.cluster_lb).mean()
        return ratio

    # ------------------------------------------------------------
    # Compression / expansion proxy
    # ------------------------------------------------------------
    def _compression(self, df: pd.DataFrame) -> pd.Series:
        rng = df["high"] - df["low"]
        rolling_mean = rng.rolling(self.comp_lb).mean()
        pct = (rng - rolling_mean).divide(rolling_mean.replace(0, np.nan))
        return pct

    # ------------------------------------------------------------
    # Toxicity estimator (volatility + wick-based whip risk)
    # ------------------------------------------------------------
    def _toxicity(self, df: pd.DataFrame) -> pd.Series:
        wicks = (df["high"] - df["close"]) + (df["close"] - df["low"])
        base = df["vol_zscore"].abs() + wicks.pct_change().abs()
        tox = base.rolling(self.tox_lb).mean()
        return tox

    # ------------------------------------------------------------
    # Projection: map 12h / 6h regime context → 15m rows
    # ------------------------------------------------------------
    def _project(self, df_15m: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
        ctx = ctx.sort_index()
        df_15m = df_15m.sort_index()
        merged = pd.merge_asof(df_15m, ctx, left_index=True, right_index=True, direction="backward")
        return merged

    # ------------------------------------------------------------
    # Build regime features on 12h TF and project → 15m
    # ------------------------------------------------------------
    def build(self, df_12h: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Construct regime feature block from 12h data, aligned onto 15m.
        """

        log("RegimeFeatures: computing volatility percentile...")
        vol_pct = self._vol_percentile(df_12h)

        log("RegimeFeatures: computing trend persistence...")
        trend_persist = self._trend_persistence(df_12h)

        log("RegimeFeatures: computing compression metric...")
        comp = self._compression(df_12h)

        log("RegimeFeatures: computing toxicity indicator...")
        tox = self._toxicity(df_12h)

        # RollingWindow lag logic for regime state transitions
        lag_cols = ["vol_pct", "trend_persist", "compression_12h", "toxicity_12h"]

        ctx = pd.DataFrame({
            "vol_pct": vol_pct,
            "trend_persist": trend_persist,
            "compression_12h": comp,
            "toxicity_12h": tox,
        }, index=df_12h.index)

        ctx = RollingWindows.add_lags(ctx, lag_cols, lags=[1, 2])

        log("RegimeFeatures: projecting 12h regime features onto 15m bars...")
        final = self._project(df_15m, ctx)

        log("RegimeFeatures: final regime feature block ready.")
        return final


class RegimeFeatureBlock:
    """
    Wrapper to build and attach regime feature block to 15m data.
    """

    def __init__(self, config_loader):
        self.cfg = config_loader
        self.regime_cfg = config_loader.load_yaml("features.yaml")["features"]["regime"]
        self.regime = RegimeFeatures(config_loader.load_yaml("features.yaml"))

        self.min_cluster_size = int(self.regime_cfg.get("hdbscan_min_cluster_size", 20))
        self.min_samples = int(self.regime_cfg.get("hdbscan_min_samples", 10))
        self.hmm_states = int(self.regime_cfg.get("hmm_states", 5))
        self.hmm_covariance = self.regime_cfg.get("hmm_covariance", "full")
        self.smoothing = float(self.regime_cfg.get("smoothing_factor", 0.7))
        self.model_clip_quantiles = self.regime_cfg.get("model_clip_quantiles", [0.005, 0.995])
        self.hdbscan_scaler = str(self.regime_cfg.get("hdbscan_scaler", "robust")).lower()
        self.hmm_scaler = str(self.regime_cfg.get("hmm_scaler", "standard")).lower()

    # --------------------------------------------------------------
    def _prep_base_regime_frame(self, df_12h: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the 12h dataframe has the minimum regime inputs.
        """
        frame = df_12h.copy()
        if "dt" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame["dt"]))
        elif "timestamp" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame["timestamp"], unit="s"))

        # Realized vol + z-score baseline
        ret = frame["close"].pct_change()
        vol = ret.rolling(30).std()
        vol_mean = vol.rolling(200).mean()
        vol_std = vol.rolling(200).std()
        frame["vol_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # Range proxies if missing
        if "range_ratio" not in frame.columns:
            rng = frame["high"] - frame["low"]
            frame["range_ratio"] = rng / frame["close"].replace(0, np.nan)

        return frame

    def _map_state_labels(self, features: pd.DataFrame, hidden_states: np.ndarray) -> Dict[int, str]:
        mapping = {}
        for state_id in np.unique(hidden_states):
            subset = features[hidden_states == state_id]
            vol = subset["vol_pct"].mean()
            trend = subset["trend_persist"].mean()
            comp = subset["compression_12h"].mean()

            if vol > 0.66 and trend > 0.55:
                label = "trend"
            elif vol > 0.66 and trend <= 0.55:
                label = "expansion"
            elif comp < -0.05:
                label = "collapse"
            else:
                label = "range"
            mapping[state_id] = label
        return mapping

    @staticmethod
    def _deterministic_regime_fallback(feat: pd.DataFrame, reg_index: pd.Index) -> pd.DataFrame:
        """
        Build deterministic regime probabilities when unsupervised models are
        unavailable or fail at runtime.
        """
        fallback = pd.DataFrame(index=feat.index)
        trend_proxy = feat["trend_persist"].clip(lower=0.0, upper=1.0)
        expansion_proxy = feat["vol_pct"].clip(lower=0.0, upper=1.0) * (1.0 - trend_proxy)
        collapse_proxy = feat["compression_12h"].lt(-0.05).astype(float) * 0.5 + feat["toxicity_12h"].clip(lower=0.0).fillna(0.0) * 0.1
        range_proxy = 1.0 + feat["compression_12h"].clip(upper=0.0).abs()
        total = trend_proxy + expansion_proxy + collapse_proxy + range_proxy
        fallback["p_regime_trend"] = (trend_proxy / total).fillna(0.25)
        fallback["p_regime_expansion"] = (expansion_proxy / total).fillna(0.25)
        fallback["p_regime_collapse"] = (collapse_proxy / total).fillna(0.25)
        fallback["p_regime_range"] = (range_proxy / total).fillna(0.25)
        fallback["regime_state"] = (
            fallback[["p_regime_trend", "p_regime_range", "p_regime_expansion", "p_regime_collapse"]]
            .idxmax(axis=1)
            .str.replace("p_regime_", "", regex=False)
        )
        return fallback.reindex(reg_index).ffill().fillna(
            {
                "p_regime_trend": 0.25,
                "p_regime_range": 0.25,
                "p_regime_expansion": 0.25,
                "p_regime_collapse": 0.25,
                "regime_state": "unknown",
            }
        )

    def _run_regime_models(self, reg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit HDBSCAN + HMM on regime features and return probability columns aligned to reg_df index.
        """
        feature_cols = ["vol_pct", "trend_persist", "compression_12h", "toxicity_12h"]
        feat = reg_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
        feat = feat.loc[feat.notna().any(axis=1)]

        if len(feat) < max(30, self.min_cluster_size):
            LOG.warning("Regime block: insufficient data for clustering; returning defaults.")
            base = pd.DataFrame(
                {
                    "p_regime_trend": 0.25,
                    "p_regime_range": 0.25,
                    "p_regime_expansion": 0.25,
                    "p_regime_collapse": 0.25,
                    "regime_state": "unknown",
                },
                index=reg_df.index,
            )
            return base

        if hdbscan is None or GaussianHMM is None:
            LOG.warning("Regime block: hdbscan/hmmlearn unavailable; returning deterministic regime defaults.")
            return self._deterministic_regime_fallback(feat, reg_df.index)

        try:
            lower_q, upper_q = self.model_clip_quantiles
            lower_q = float(min(max(float(lower_q), 0.0), 0.49))
            upper_q = float(max(min(float(upper_q), 1.0), 0.51))
            fill_values = feat.median(axis=0, numeric_only=True).fillna(0.0)
            clip_lower = feat.quantile(lower_q, interpolation="linear").fillna(fill_values)
            clip_upper = feat.quantile(upper_q, interpolation="linear").fillna(fill_values)
            feat = feat.fillna(fill_values).clip(lower=clip_lower, upper=clip_upper, axis=1).fillna(0.0)

            hdb_scaler = StandardScaler() if self.hdbscan_scaler == "standard" else RobustScaler()
            hmm_scaler = RobustScaler() if self.hmm_scaler == "robust" else StandardScaler()
            X_hdb = hdb_scaler.fit_transform(feat[feature_cols].values)
            X_hmm = hmm_scaler.fit_transform(feat[feature_cols].values)

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                allow_single_cluster=True,
            )
            cluster_labels = clusterer.fit_predict(X_hdb)
            feat = feat.assign(cluster=cluster_labels)

            # HMM smoothing on feature vectors
            n_states = max(2, min(self.hmm_states, len(feat) - 1))
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type=self.hmm_covariance,
                n_iter=50,
                verbose=False,
            )
            hmm.fit(X_hmm)
            posterior = hmm.predict_proba(X_hmm)
            hidden_seq = hmm.predict(X_hmm)
        except Exception as exc:
            LOG.warning(
                "Regime block: model fit failed (%s); using deterministic fallback.",
                exc,
            )
            return self._deterministic_regime_fallback(feat, reg_df.index)

        state_map = self._map_state_labels(feat.assign(hidden=hidden_seq), hidden_seq)

        prob_df = pd.DataFrame(
            {
                "p_regime_trend": 0.0,
                "p_regime_range": 0.0,
                "p_regime_expansion": 0.0,
                "p_regime_collapse": 0.0,
                "regime_state": "unknown",
            },
            index=feat.index,
        )

        for idx, probs in zip(feat.index, posterior):
            trend_p = 0.0
            range_p = 0.0
            exp_p = 0.0
            coll_p = 0.0
            for state_id, p in enumerate(probs):
                label = state_map.get(state_id, "range")
                if label == "trend":
                    trend_p += p
                elif label == "expansion":
                    exp_p += p
                elif label == "collapse":
                    coll_p += p
                else:
                    range_p += p
            prob_df.loc[idx, "p_regime_trend"] = trend_p
            prob_df.loc[idx, "p_regime_range"] = range_p
            prob_df.loc[idx, "p_regime_expansion"] = exp_p
            prob_df.loc[idx, "p_regime_collapse"] = coll_p
            # hard label
            max_label = max(
                [("trend", trend_p), ("range", range_p), ("expansion", exp_p), ("collapse", coll_p)],
                key=lambda t: t[1],
            )[0]
            prob_df.loc[idx, "regime_state"] = max_label

        prob_df = prob_df.reindex(reg_df.index).ffill()
        return prob_df

    def apply(self, df_15m: pd.DataFrame, df_6h: pd.DataFrame, df_12h: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Build regime features from 12h data, project to 15m, and attach HDBSCAN+HMM probabilities.
        """
        if df_15m is None or df_15m.empty or df_12h is None or df_12h.empty:
            return df_15m

        df12_base = self._prep_base_regime_frame(df_12h)
        df15_idx = df_15m.copy()
        if "dt" in df15_idx.columns:
            df15_idx = df15_idx.set_index(pd.to_datetime(df15_idx["dt"]))
        elif "timestamp" in df15_idx.columns:
            df15_idx = df15_idx.set_index(pd.to_datetime(df15_idx["timestamp"], unit="s"))

        # Build regime feature block and project onto 15m
        reg_features = self.regime.build(df12_base, df15_idx)

        # Fit clustering + HMM on the regime feature slice
        prob_df = self._run_regime_models(reg_features)

        merged = reg_features.join(prob_df, how="left")
        state_map = {"trend": 1, "range": 0, "expansion": 2, "collapse": -1, "unknown": 0}
        merged["regime_state_id"] = merged["regime_state"].map(state_map).fillna(0).astype(int)
        if "dt" not in merged.columns:
            merged["dt"] = pd.to_datetime(merged.index, utc=True)
        merged["timestamp"] = (
            pd.to_datetime(merged["dt"], utc=True).astype("int64") // 10**9
        ).astype("int64")
        LOG.info("RegimeFeatureBlock applied.")
        return merged.reset_index(drop=True)


# Backward-compatible alias expected by callers
RegimeFeatureBuilder = RegimeFeatureBlock
