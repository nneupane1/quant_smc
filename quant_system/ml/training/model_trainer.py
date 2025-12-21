"""
Hyperparameter-driven, walk-forward trainer for all ML components.
Implements:
 - Specialist classifiers with Optuna HPO (LightGBM/XGBoost/Logistic)
 - Probability calibration (platt/isotonic/empirical)
 - Meta + Confluence stacking
 - Hazard per-bin logistic models
 - Quantile return forecaster
 - Production-grade preprocessing (impute/scale + OHE)
"""

from typing import Dict, List, Any, Tuple, Optional
import time

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier

import lightgbm as lgb
import xgboost as xgb

from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.predict.empirical_calibrator import EmpiricalCalibrator
from quant_system.ml.registry.versioning import ModelVersionManager

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("model_trainer")


class ModelTrainer:
    """
    Full walk-forward trainer for all ML specialists across all assets.
    Config-driven algorithms, HPO ranges, CV folds, and calibrators.
    """

    def __init__(self, config_loader: ConfigLoader, registry: ModelRegistry):
        self.cfg_loader = config_loader
        self.model_cfg = config_loader.load_yaml("models.yaml")["models"]
        self.assets_cfg = config_loader.load_yaml("assets.yaml")
        self.labels_cfg = config_loader.load_yaml("labels.yaml")
        self.features_cfg = config_loader.load_yaml("features.yaml")

        self.registry = registry
        self.versioner = ModelVersionManager()

        LOG.info("[ModelTrainer] Initialized (walk-forward + HPO)")

    # ------------------------------------------------------------------
    # Utility: class weights and single-class safety
    # ------------------------------------------------------------------
    def _make_class_weight(self, y: np.ndarray, cfg: Dict[str, Any]) -> Tuple[Optional[Dict[int, float]], Optional[float]]:
        """Compute class_weight dict and scale_pos_weight for tree models."""
        series = pd.Series(y)
        counts = series.value_counts()
        if counts.empty:
            return None, None
        n_pos = float(counts.get(1, 0.0))
        n_neg = float(counts.get(0, 0.0))
        if n_pos == 0 or n_neg == 0:
            return None, None

        # explicit override from YAML wins
        cw_cfg = cfg.get("class_weight")
        if cw_cfg:
            return cw_cfg, (n_neg / n_pos if n_pos > 0 else None)

        total = n_pos + n_neg
        cw = {
            0: total / (2.0 * n_neg),
            1: total / (2.0 * n_pos),
        }
        scale_pos = n_neg / n_pos
        return cw, scale_pos

    # ------------------------------------------------------------------
    # Public entry: train all models for an asset
    # ------------------------------------------------------------------
    def train_asset(self, df: pd.DataFrame, asset: str) -> str:
        LOG.info(f"[ModelTrainer] Training full model suite for asset={asset}")
        t0 = time.perf_counter()

        targets = {
            "liq_flow": df["label_liq_flow"].astype(int).values,
            "bos_cont": df["label_bos_cont"].astype(int).values,
            "momo": df["label_momo"].astype(int).values,
            "eop": df["label_eop"].astype(int).values,
            "edp": df["label_edp"].astype(int).values,
        }

        haz_event = df["hazard_event"].astype(int).values
        haz_time = df["hazard_time"].astype(int).values
        price_col = None
        for cand in ("close", "close_x", "close_y"):
            if cand in df.columns:
                price_col = cand
                break
        if price_col is None:
            raise KeyError("No close/close_x/close_y column found for price extraction.")
        prices = df[price_col].astype(float).values

        # Specialist models ------------------------------------------------
        specialists = {}
        specialist_metrics = {}
        for key in ["liq_flow", "bos_cont", "momo", "eop", "edp"]:
            cfg = self.model_cfg[key]
            X_sel, cols_sel, num_cols, cat_cols = self._prepare_features(df, groups=cfg.get("features", {}).get("group"))
            LOG.info(f"[ModelTrainer] Specialist {key} | features={len(cols_sel)}")
            t_spec = time.perf_counter()
            model, metrics = self._train_classifier(
                X_sel, targets[key], cfg, name=f"{asset}_{key}", num_cols=num_cols, cat_cols=cat_cols
            )
            model = self._calibrate(model, X_sel, targets[key], cfg.get("calibrator"))
            specialists[key] = {"model": model, "feature_cols": cols_sel}
            specialist_metrics[key] = metrics
            LOG.info(f"[ModelTrainer] Specialist {key} done in {time.perf_counter() - t_spec:.2f}s")

        # Meta model (stacking specialist outputs)
        meta_cfg = self.model_cfg.get("meta_model", {})
        LOG.info("[ModelTrainer] Training meta model (stacking)")
        meta_model = self._train_stack(df, specialists, meta_cfg, target_key="label_liq_flow")
        LOG.info("[ModelTrainer] Meta model done")

        # Confluence model
        conf_cfg = self.model_cfg.get("confluence_model", {})
        LOG.info("[ModelTrainer] Training confluence model")
        conf_model = self._train_stack(df, specialists, conf_cfg, target_key="label_liq_flow")
        LOG.info("[ModelTrainer] Confluence model done")

        # Hazard per-bin model
        haz_cfg = self.model_cfg.get("hazard", {})
        LOG.info("[ModelTrainer] Training hazard models")
        hazard_models, haz_conf = self._train_hazard(df, haz_cfg, haz_event, haz_time)
        LOG.info(f"[ModelTrainer] Hazard models done ({len(hazard_models)} bins)")

        # Quantile forecaster
        q_cfg = self.model_cfg.get("quantile_forecaster", {})
        LOG.info("[ModelTrainer] Training quantile forecaster")
        quant_models = self._train_quantile(df, q_cfg, prices)
        LOG.info("[ModelTrainer] Quantile forecaster done")

        # Persist
        version = self.versioner.new_version(asset)
        for key, bundle in specialists.items():
            self.registry.save_model(
                model_name=f"{asset}_{key}",
                version=version,
                clf=bundle["model"],
                cal=None,
                config={"features": bundle["feature_cols"]},
            )
            # generic alias for single-asset deployments
            self.registry.save_model(
                model_name=key,
                version=version,
                clf=bundle["model"],
                cal=None,
                config={"features": bundle["feature_cols"]},
            )
            # metrics (cv and params)
            metrics = specialist_metrics.get(key, {})
            if metrics:
                self.registry.save_metrics(f"{asset}_{key}", version, metrics)
                self.registry.save_metrics(key, version, metrics)

        self.registry.save_model(
            model_name=f"{asset}_meta",
            version=version,
            clf=meta_model,
            cal=None,
            config={},
        )
        self.registry.save_model(
            model_name="meta_model",
            version=version,
            clf=meta_model,
            cal=None,
            config={},
        )
        self.registry.save_model(
            model_name=f"{asset}_confluence",
            version=version,
            clf=conf_model,
            cal=None,
            config={},
        )
        self.registry.save_model(
            model_name="confluence_model",
            version=version,
            clf=conf_model,
            cal=None,
            config={},
        )

        self.registry.save_hazard_model(
            model_name=f"{asset}_hazard",
            version=version,
            models=hazard_models,
            config=haz_conf,
        )
        self.registry.save_hazard_model(
            model_name="hazard",
            version=version,
            models=hazard_models,
            config=haz_conf,
        )

        self.registry.save_model(
            model_name=f"{asset}_quantile",
            version=version,
            clf=quant_models,
            cal=None,
            config={"quantiles": q_cfg.get("quantiles", [])},
        )
        self.registry.save_model(
            model_name="quantile",
            version=version,
            clf=quant_models,
            cal=None,
            config={"quantiles": q_cfg.get("quantiles", [])},
        )

        LOG.info(f"[ModelTrainer] Completed asset={asset}, version={version}, elapsed={time.perf_counter()-t0:.2f}s")
        return version

    # ------------------------------------------------------------------
    # Feature selection helpers
    # ------------------------------------------------------------------
    def _prepare_features(
        self, df: pd.DataFrame, groups: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
        """
        Select feature columns by configured groups; drop labels/timestamps.
        Returns (feature_df, feature_cols, numeric_cols, categorical_cols).
        """
        drop_cols = {
            "dt",
            "timestamp",
            "label_liq_flow",
            "label_bos_cont",
            "label_momo",
            "label_eop",
            "label_edp",
            "hazard_event",
            "hazard_time",
        }
        all_feature_cols = [c for c in df.columns if c not in drop_cols]

        if not groups:
            use_cols = all_feature_cols
        else:
            prefixes = {
                "smc": ("swing_", "bos_", "fvg_", "sweep_", "demand_", "supply_", "compression_", "pd_", "zone_"),
                "ema": ("ema", "band", "sma", "hma"),
                "volatility": ("atr", "vol_", "rng_", "range_", "rstd", "z"),
                "liquidity": ("liq_", "wick", "eql", "liq"),
                "regime": ("p_regime", "regime", "vol_pct", "trend_persist", "compression_12h", "toxicity"),
                "session": ("session_",),
            }
            use_cols = []
            for col in all_feature_cols:
                for g in groups:
                    if any(col.startswith(p) for p in prefixes.get(g, ())):
                        use_cols.append(col)
                        break
            if not use_cols:
                use_cols = all_feature_cols

        X_df = df[use_cols].copy()

        # Drop columns that are entirely NaN or constant to avoid downstream imputer warnings
        dropped: List[str] = []
        for col in list(X_df.columns):
            series = X_df[col]
            if not series.notna().any():
                dropped.append(col)
                X_df.drop(columns=[col], inplace=True)
                continue
            if series.dropna().nunique() <= 1:
                dropped.append(col)
                X_df.drop(columns=[col], inplace=True)

        if dropped:
            LOG.warning(f"[ModelTrainer] Dropped non-informative columns: {dropped}")

        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]
        return X_df, list(X_df.columns), num_cols, cat_cols

    def _build_preprocessor(self, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
            remainder="drop",
        )

    # ------------------------------------------------------------------
    # Core classifier training with HPO + CV
    # ------------------------------------------------------------------
    def _train_classifier(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        cfg: Dict[str, Any],
        name: str,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> Tuple[Any, Dict[str, Any]]:
        algo = cfg.get("algorithm", "lightgbm").lower()
        n_trials = int(cfg.get("hpo_trials", 30))
        cv_splits = int(cfg.get("cv_splits", 4))
        hpo_space = cfg.get("hpo_space", {})

        # handle imbalance & single-class upfront
        cw, scale_pos = self._make_class_weight(y, cfg)
        class_counts = pd.Series(y).value_counts().to_dict()
        if cw:
            LOG.info(f"[ModelTrainer] {name} class_weight={cw} counts={class_counts}")
        else:
            LOG.info(f"[ModelTrainer] {name} class counts={class_counts} (no weighting applied)")
        # single-class guard: fall back to a constant predictor so pipeline doesn't explode
        if pd.Series(y).nunique() < 2:
            LOG.warning(f"[ModelTrainer] {name} target has single class; training DummyClassifier.")
            pre = self._build_preprocessor(num_cols, cat_cols)
            dummy = DummyClassifier(strategy="most_frequent")
            model = Pipeline([("pre", pre), ("clf", dummy)])
            model.fit(X_df, y)
            metrics = {"cv_score": None, "best_params": {}, "hpo_trials": 0, "class_counts": class_counts}
            return model, metrics

        LOG.info(f"[ModelTrainer] HPO for {name} algo={algo} trials={n_trials} splits={cv_splits}")
        hpo_t0 = time.perf_counter()

        def build_model(params: Dict[str, Any]):
            if algo == "lightgbm":
                return lgb.LGBMClassifier(
                    n_estimators=int(params["n_estimators"]),
                    num_leaves=int(params["num_leaves"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    subsample=float(params.get("subsample", 1.0)),
                    colsample_bytree=float(params.get("colsample_bytree", 1.0)),
                    reg_alpha=float(params.get("reg_alpha", 0.0)),
                    reg_lambda=float(params.get("reg_lambda", 0.0)),
                    objective="binary",
                    class_weight=cw,
                )
            if algo == "xgboost":
                return xgb.XGBClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    subsample=float(params.get("subsample", 1.0)),
                    colsample_bytree=float(params.get("colsample_bytree", 1.0)),
                    reg_alpha=float(params.get("reg_alpha", 0.0)),
                    reg_lambda=float(params.get("reg_lambda", 0.0)),
                    eval_metric="logloss",
                    tree_method="hist",
                    objective="binary:logistic",
                    scale_pos_weight=scale_pos,
                )
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=500,
                penalty="l2",
                solver="lbfgs",
                class_weight=cw,
            )

        def param_space(trial: optuna.Trial) -> Dict[str, Any]:
            if algo in ("lightgbm", "xgboost"):
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", int(hpo_space.get("n_estimators", [200, 600])[0]),
                        int(hpo_space.get("n_estimators", [200, 600])[1])
                    ),
                    "num_leaves": trial.suggest_int(
                        "num_leaves", int(hpo_space.get("num_leaves", [31, 255])[0]),
                        int(hpo_space.get("num_leaves", [31, 255])[1])
                    ) if algo == "lightgbm" else None,
                    "max_depth": trial.suggest_int(
                        "max_depth", int(hpo_space.get("max_depth", [3, 12])[0]),
                        int(hpo_space.get("max_depth", [3, 12])[1])
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", float(hpo_space.get("learning_rate", [0.001, 0.15])[0]),
                        float(hpo_space.get("learning_rate", [0.001, 0.15])[1]),
                        log=True,
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", float(hpo_space.get("subsample", [0.6, 1.0])[0]),
                        float(hpo_space.get("subsample", [0.6, 1.0])[1]),
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", float(hpo_space.get("colsample_bytree", [0.6, 1.0])[0]),
                        float(hpo_space.get("colsample_bytree", [0.6, 1.0])[1]),
                    ),
                    "reg_alpha": trial.suggest_float(
                        "reg_alpha", float(hpo_space.get("reg_alpha", [0.0, 10.0])[0]),
                        float(hpo_space.get("reg_alpha", [0.0, 10.0])[1]),
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", float(hpo_space.get("reg_lambda", [0.0, 10.0])[0]),
                        float(hpo_space.get("reg_lambda", [0.0, 10.0])[1]),
                    ),
                }
            return {
                "C": trial.suggest_float(
                    "C", float(hpo_space.get("C", [0.01, 10.0])[0]),
                    float(hpo_space.get("C", [0.01, 10.0])[1]),
                    log=True,
                )
            }

        def ts_score(params: Dict[str, Any]) -> float:
            base_model = build_model(params)
            pre = self._build_preprocessor(num_cols, cat_cols)
            model = Pipeline([("pre", pre), ("clf", base_model)])
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            for tr_idx, va_idx in tscv.split(X_df):
                Xt, Xv = X_df.iloc[tr_idx], X_df.iloc[va_idx]
                yt, yv = y[tr_idx], y[va_idx]
                model.fit(Xt, yt)
                prob = model.predict_proba(Xv)[:, 1]
                pr = average_precision_score(yv, prob)
                try:
                    auc = roc_auc_score(yv, prob)
                except Exception:
                    auc = 0.5
                scores.append(pr if not np.isnan(pr) else auc)
            return float(np.mean(scores))

        def objective(trial: optuna.Trial):
            params = param_space(trial)
            params = {k: v for k, v in params.items() if v is not None}
            return -ts_score(params)

        def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            # periodic progress logging
            if trial.value is None:
                return
            if (trial.number % 5 == 0) or (trial.number + 1 == n_trials):
                best = study.best_value
                LOG.info(
                    f"[ModelTrainer] {name} HPO trial {trial.number + 1}/{n_trials} "
                    f"value={trial.value:.4f} best={best:.4f} elapsed={time.perf_counter() - hpo_t0:.1f}s"
                )

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_cb])
        best_params = study.best_params
        best_cv = -study.best_value
        LOG.info(
            f"[ModelTrainer] Best params {name}: {best_params} | cv_score={best_cv:.4f} "
            f"HPO elapsed={time.perf_counter() - hpo_t0:.1f}s"
        )

        best_params = {k: v for k, v in best_params.items() if v is not None}
        pre = self._build_preprocessor(num_cols, cat_cols)
        final_model = Pipeline([("pre", pre), ("clf", build_model(best_params))])
        final_model.fit(X_df, y)
        metrics = {
            "cv_score": best_cv,
            "best_params": best_params,
            "hpo_trials": n_trials,
            # record label balance for downstream monitoring
            "class_counts": class_counts,
        }
        return final_model, metrics

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _calibrate(self, model, X_df: pd.DataFrame, y: np.ndarray, method: str):
        if not hasattr(model, "predict_proba"):
            return model
        # If target has a single class, skip calibration gracefully
        if pd.Series(y).nunique() < 2:
            LOG.warning("[ModelTrainer] Skipping calibration: only one class present in target.")
            return model

        method = (method or "auto").lower()
        if method in ("platt", "isotonic"):
            cv = max(3, min(5, len(y) // 200))
            calib = CalibratedClassifierCV(model, cv=cv, method="sigmoid" if method == "platt" else "isotonic")
            calib.fit(X_df, y)
            return calib

        p_raw = model.predict_proba(X_df)[:, 1]
        # If probabilities are degenerate, skip calibration
        if np.unique(p_raw).size < 2:
            LOG.warning("[ModelTrainer] Skipping calibration: predicted probabilities are constant.")
            cal = None
        else:
            cal = EmpiricalCalibrator(method="auto").calibrate(p_raw, y)

        class CalibratedModel:
            def __init__(self, base, calibrator):
                self.base = base
                self.cal = calibrator

            def predict_proba(self, X_in):
                p = self.base.predict_proba(X_in)[:, 1]
                if self.cal:
                    if hasattr(self.cal, "predict_proba"):
                        p = self.cal.predict_proba(p.reshape(-1, 1))[:, 1]
                    elif hasattr(self.cal, "predict"):
                        p = self.cal.predict(p)
                    else:
                        p = np.array([self.cal(pi) for pi in p])
                return np.vstack([1 - p, p]).T

        return CalibratedModel(model, cal)

    # ------------------------------------------------------------------
    # Stacking (meta / confluence)
    # ------------------------------------------------------------------
    def _train_stack(self, df: pd.DataFrame, specialists: Dict[str, Dict[str, Any]], cfg: Dict[str, Any], target_key: str):
        inputs = cfg.get("specialist_inputs", list(specialists.keys()))
        hpo_space = cfg.get("hpo_space", {})
        n_trials = int(cfg.get("hpo_trials", 10))
        cv_splits = int(cfg.get("cv_splits", 3))

        base_cols = []
        for key in inputs:
            model = specialists[key]["model"]
            cols = specialists[key]["feature_cols"]
            base_cols.append((key, model, cols))

        X_meta_parts = []
        y_meta = df[target_key].astype(int).values if target_key in df else df["label_liq_flow"].astype(int).values
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        for train_idx, val_idx in tscv.split(df):
            fold_preds = []
            for _, model, cols in base_cols:
                X_fold = df.iloc[val_idx][cols]
                prob = model.predict_proba(X_fold)[:, 1]
                fold_preds.append(prob)
            X_fold_mat = np.vstack(fold_preds).T
            X_meta_parts.append((val_idx, X_fold_mat))

        meta_mat = np.zeros((len(df), len(base_cols)))
        for val_idx, mat in X_meta_parts:
            meta_mat[val_idx, :] = mat

        def build_meta(params: Dict[str, Any]):
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=500,
                penalty="l2",
                solver="lbfgs",
            )

        def param_space(trial: optuna.Trial) -> Dict[str, Any]:
            return {
                "C": trial.suggest_float(
                    "C", float(hpo_space.get("C", [0.01, 5.0])[0]),
                    float(hpo_space.get("C", [0.01, 5.0])[1]),
                    log=True,
                )
            }

        def meta_score(params: Dict[str, Any]) -> float:
            model = build_meta(params)
            scores = []
            for tr_idx, va_idx in tscv.split(meta_mat):
                Xt, Xv = meta_mat[tr_idx], meta_mat[va_idx]
                yt, yv = y_meta[tr_idx], y_meta[va_idx]
                model.fit(Xt, yt)
                prob = model.predict_proba(Xv)[:, 1]
                pr = average_precision_score(yv, prob)
                scores.append(pr)
            return float(np.mean(scores))

        def objective(trial: optuna.Trial):
            return -meta_score(param_space(trial))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params

        final_meta = build_meta(best_params)
        final_meta.fit(meta_mat, y_meta)
        LOG.info(f"[ModelTrainer] Stacking model trained with params={best_params}")
        return final_meta

    # ------------------------------------------------------------------
    # Hazard model (per-bin logistic)
    # ------------------------------------------------------------------
    def _train_hazard(self, df: pd.DataFrame, cfg: Dict[str, Any], event: np.ndarray, tte: np.ndarray):
        horizon = int(cfg.get("horizon_bars", 48))
        hpo_trials = int(cfg.get("hpo", {}).get("trials_per_bin", 5))
        cv_splits = int(cfg.get("cv_splits", 3))
        hpo_space = cfg.get("hpo_space", {"C": [0.01, 10.0]})

        X_all, feature_cols, num_cols, cat_cols = self._prepare_features(df, groups=cfg.get("features", {}).get("group"))
        models = {}

        for b in range(1, horizon + 1):
            y_bin = ((event == 1) & (tte <= b)).astype(int)
            if y_bin.mean() < cfg.get("min_event_fraction", 0.005):
                continue

            def param_space(trial: optuna.Trial):
                return {
                    "C": trial.suggest_float(
                        "C", float(hpo_space.get("C", [0.01, 10.0])[0]),
                        float(hpo_space.get("C", [0.01, 10.0])[1]),
                        log=True,
                    )
                }

            def objective(trial: optuna.Trial):
                params = param_space(trial)
                base = LogisticRegression(C=float(params["C"]), max_iter=300, penalty="l2", solver="lbfgs")
                pre = self._build_preprocessor(num_cols, cat_cols)
                model = Pipeline([("pre", pre), ("clf", base)])
                tscv = TimeSeriesSplit(n_splits=cv_splits)
                scores = []
                for tr_idx, va_idx in tscv.split(X_all):
                    Xt, Xv = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                    yt, yv = y_bin[tr_idx], y_bin[va_idx]
                    model.fit(Xt, yt)
                    prob = model.predict_proba(Xv)[:, 1]
                    pr = average_precision_score(yv, prob)
                    scores.append(pr)
                return -float(np.mean(scores))

            if cfg.get("hpo", {}).get("enabled", True):
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=hpo_trials, show_progress_bar=False)
                best_C = study.best_params["C"]
            else:
                best_C = 1.0

            clf = LogisticRegression(C=float(best_C), max_iter=400, penalty="l2", solver="lbfgs")
            pre = self._build_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_all, y_bin)
            models[b] = pipe

        LOG.info(f"[ModelTrainer] Hazard model trained with {len(models)} bins.")
        return models, {"horizon_bars": horizon, "feature_cols": feature_cols}

    # ------------------------------------------------------------------
    # Quantile forecaster (LightGBM regressor)
    # ------------------------------------------------------------------
    def _train_quantile(self, df: pd.DataFrame, cfg: Dict[str, Any], prices: np.ndarray):
        quantiles = cfg.get("quantiles", [0.05, 0.1, 0.5, 0.9, 0.95])
        n_trials = int(cfg.get("hpo_trials", 20))
        cv_splits = int(cfg.get("cv_splits", 3))
        hpo_space = cfg.get("hpo_space", {})

        returns = np.diff(prices, prepend=prices[0])
        X_all, feature_cols, num_cols, cat_cols = self._prepare_features(df, groups=cfg.get("features", {}).get("group"))

        models = {}

        for q in quantiles:
            LOG.info(f"[ModelTrainer] Training quantile model q={q}")

            def param_space(trial: optuna.Trial):
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", int(hpo_space.get("n_estimators", [200, 600])[0]),
                        int(hpo_space.get("n_estimators", [200, 600])[1])
                    ),
                    "num_leaves": trial.suggest_int(
                        "num_leaves", int(hpo_space.get("num_leaves", [31, 255])[0]),
                        int(hpo_space.get("num_leaves", [31, 255])[1])
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth", int(hpo_space.get("max_depth", [3, 12])[0]),
                        int(hpo_space.get("max_depth", [3, 12])[1])
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", float(hpo_space.get("learning_rate", [0.001, 0.15])[0]),
                        float(hpo_space.get("learning_rate", [0.001, 0.15])[1]),
                        log=True,
                    ),
                    "subsample": trial.suggest_float(
                        "subsample", float(hpo_space.get("subsample", [0.6, 1.0])[0]),
                        float(hpo_space.get("subsample", [0.6, 1.0])[1]),
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", float(hpo_space.get("colsample_bytree", [0.6, 1.0])[0]),
                        float(hpo_space.get("colsample_bytree", [0.6, 1.0])[1]),
                    ),
                }

            def objective(trial: optuna.Trial):
                params = param_space(trial)
                base_model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=int(params["n_estimators"]),
                    num_leaves=int(params["num_leaves"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    subsample=float(params["subsample"]),
                    colsample_bytree=float(params["colsample_bytree"]),
                )
                pre = self._build_preprocessor(num_cols, cat_cols)
                model = Pipeline([("pre", pre), ("reg", base_model)])
                tscv = TimeSeriesSplit(n_splits=cv_splits)
                losses = []
                for tr_idx, va_idx in tscv.split(X_all):
                    Xt, Xv = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                    yt, yv = returns[tr_idx], returns[va_idx]
                    model.fit(Xt, yt)
                    pred = model.predict(Xv)
                    e = yv - pred
                    pinball = np.mean(np.maximum(q * e, (q - 1) * e))
                    losses.append(pinball)
                return float(np.mean(losses))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params

            base_final = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=int(best_params["n_estimators"]),
                num_leaves=int(best_params["num_leaves"]),
                max_depth=int(best_params["max_depth"]),
                learning_rate=float(best_params["learning_rate"]),
                subsample=float(best_params["subsample"]),
                colsample_bytree=float(best_params["colsample_bytree"]),
            )
            pre = self._build_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([("pre", pre), ("reg", base_final)])
            pipe.fit(X_all, returns)
            models[f"q_{q}"] = pipe

        return models
