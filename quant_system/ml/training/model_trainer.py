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
from copy import deepcopy
from contextlib import contextmanager
import time
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.base import BaseEstimator, TransformerMixin
from quant_system.utils.pandas_compat import ensure_stringmethods_alias

try:
    import optuna
except ImportError:  # pragma: no cover - runtime fallback
    optuna = None

try:
    ensure_stringmethods_alias()
    import lightgbm as lgb
except Exception:  # pragma: no cover - runtime fallback
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - runtime fallback
    xgb = None

from quant_system.ml.registry.model_registry import ModelRegistry
from quant_system.ml.predict.empirical_calibrator import EmpiricalCalibrator
from quant_system.ml.registry.versioning import ModelVersionManager

from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import (
    console_kv,
    console_stage,
    fmt_num,
    fmt_seconds,
    get_logger,
)

LOG = get_logger("model_trainer")


class CalibratedProbabilityModel:
    """
    Pickle-friendly probability wrapper for empirical calibration.
    """

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


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Leak-safe numeric outlier clipping.
    Learns per-feature quantile bounds on training folds only, then clips.
    """

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.lower_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.lower_ is None or self.upper_ is None:
            return arr
        return np.clip(arr, self.lower_, self.upper_)


class ModelTrainer:
    """
    Full walk-forward trainer for all ML specialists across all assets.
    Config-driven algorithms, HPO ranges, CV folds, and calibrators.
    """

    PREDICTION_OUTPUTS = {
        "liq_flow",
        "bos_cont",
        "flow_1h",
        "momo",
        "eop",
        "edp",
        "meta",
        "confluence",
        "hazard",
    }
    SPECIALIST_MODELS = ["liq_flow", "bos_cont", "flow_1h", "momo", "eop", "edp"]
    TRAINABLE_MODELS = SPECIALIST_MODELS + ["meta_model", "confluence_model", "hazard", "quantile"]
    MODEL_NAME_ALIASES = {
        "meta": "meta_model",
        "confluence": "confluence_model",
        "quantile_forecaster": "quantile",
        "quantile_model": "quantile",
    }

    def __init__(self, config_loader: ConfigLoader, registry: ModelRegistry):
        self.cfg_loader = config_loader
        models_yaml = config_loader.load_yaml("models.yaml")
        self.model_cfg = models_yaml["models"]
        self.preproc_cfg = models_yaml.get("training_preprocessing", {})
        self.training_log_cfg = deepcopy(models_yaml.get("training_logging", {}))
        self.heartbeat_seconds = max(int(self.training_log_cfg.get("heartbeat_seconds", 60)), 5)
        self.feature_sel_cfg = deepcopy(self.preproc_cfg.get("feature_selection", {}))
        self.optim_cfg = config_loader.load_yaml("optimization.yaml").get("optimization", {})
        self.assets_cfg = config_loader.load_yaml("assets.yaml")
        self.labels_cfg = config_loader.load_yaml("labels.yaml")
        self.features_cfg = config_loader.load_yaml("features.yaml")

        self.registry = registry
        version_index = Path(getattr(registry, "base_dir", ".")) / ".model_versions.json"
        self.versioner = ModelVersionManager(str(version_index))

        LOG.info("[ModelTrainer] Initialized (walk-forward + HPO)")

    @staticmethod
    @contextmanager
    def _suppress_low_signal_warnings():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="No positive class found in y_true.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Only one class is present in y_true.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="The least populated class in y has only.*",
                category=UserWarning,
            )
            if optuna is not None:
                try:
                    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                except Exception:
                    pass
            yield

    @staticmethod
    def _display_name(name: str) -> str:
        return str(name).replace("_", " ").upper()

    def _log_bundle_progress(
        self,
        *,
        asset: str,
        completed: int,
        total: int,
        phase: str,
        started_at: float,
    ) -> None:
        pct = (100.0 * completed / max(total, 1))
        elapsed = time.perf_counter() - started_at
        rate = completed / max(elapsed, 1e-6)
        remaining = max(total - completed, 0)
        eta = remaining / max(rate, 1e-6)
        console_stage(
            "Training Progress",
            (
                f"{asset} {completed}/{total} ({pct:.1f}%) "
                f"phase={phase} elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
            ),
            status="info",
        )

    def _log_training_header(
        self,
        name: str,
        algo: str,
        rows: int,
        features: int,
        class_counts: Optional[Dict[Any, Any]] = None,
        trials: Optional[int] = None,
    ) -> None:
        payload = {
            "model": self._display_name(name),
            "algo": algo,
            "rows": fmt_num(rows),
            "features": fmt_num(features),
        }
        if trials is not None:
            payload["trials"] = fmt_num(trials)
        if class_counts is not None:
            payload["class_mix"] = ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))
        console_kv("Model Room", payload, style="magenta")

    def _log_hpo_progress(
        self,
        name: str,
        trial_number: int,
        total_trials: int,
        trial_value: float,
        best_value: float,
        started_at: float,
        direction: str = "maximize",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if direction == "minimize":
            current_display = trial_value
            best_display = best_value
        else:
            current_display = -trial_value
            best_display = -best_value
        payload = {
            "model": self._display_name(name),
            "trial": f"{trial_number}/{total_trials}",
            "score": f"{current_display:.4f}",
            "best": f"{best_display:.4f}",
            "elapsed": fmt_seconds(time.perf_counter() - started_at),
        }
        if extra:
            payload.update(extra)
        console_kv("HPO Pulse", payload, style="bright_blue")

    def _log_hpo_summary(
        self,
        name: str,
        best_score: float,
        best_params: Dict[str, Any],
        started_at: float,
        metric_label: str = "best_score",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "model": self._display_name(name),
            metric_label: f"{best_score:.4f}",
            "elapsed": fmt_seconds(time.perf_counter() - started_at),
            "params": best_params,
        }
        if extra:
            payload.update(extra)
        console_kv("HPO Summary", payload, style="green")

    @staticmethod
    def _midpoint(bounds, cast=float):
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            low, high = bounds[0], bounds[1]
            value = (float(low) + float(high)) / 2.0
        else:
            value = bounds
        if cast is int:
            return int(round(float(value)))
        return cast(value)

    @staticmethod
    def _default_params_for_algo(algo: str, hpo_space: Dict[str, Any]) -> Dict[str, Any]:
        if algo in ("lightgbm", "xgboost", "gbr"):
            params = {
                "n_estimators": ModelTrainer._midpoint(hpo_space.get("n_estimators", [200, 600]), int),
                "max_depth": ModelTrainer._midpoint(hpo_space.get("max_depth", [3, 12]), int),
                "learning_rate": ModelTrainer._midpoint(hpo_space.get("learning_rate", [0.001, 0.15]), float),
                "subsample": ModelTrainer._midpoint(hpo_space.get("subsample", [0.6, 1.0]), float),
                "colsample_bytree": ModelTrainer._midpoint(hpo_space.get("colsample_bytree", [0.6, 1.0]), float),
                "reg_alpha": ModelTrainer._midpoint(hpo_space.get("reg_alpha", [0.0, 10.0]), float),
                "reg_lambda": ModelTrainer._midpoint(hpo_space.get("reg_lambda", [0.0, 10.0]), float),
            }
            if algo == "lightgbm":
                params["num_leaves"] = ModelTrainer._midpoint(hpo_space.get("num_leaves", [31, 255]), int)
            return params
        return {"C": ModelTrainer._midpoint(hpo_space.get("C", [0.01, 10.0]), float)}

    def _make_study(self, direction: str, cfg: Dict[str, Any]):
        if optuna is None:
            return None
        sampler_name = str(
            cfg.get("hpo_sampler")
            or (self.optim_cfg.get("bayesian", {}) or {}).get("sampler", "tpe")
        ).lower()
        bayes_cfg = self.optim_cfg.get("bayesian", {}) or {}
        seed = int(bayes_cfg.get("seed", 42))

        try:
            if sampler_name == "random":
                sampler = optuna.samplers.RandomSampler(seed=seed)
            else:
                sampler = optuna.samplers.TPESampler(
                    seed=seed,
                    multivariate=bool(bayes_cfg.get("multivariate", False)),
                )
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=int(bayes_cfg.get("pruner_startup_trials", 5)),
                n_warmup_steps=int(bayes_cfg.get("pruner_warmup_steps", 0)),
            )
            return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
        except Exception:
            return optuna.create_study(direction=direction)

    @staticmethod
    def _is_int_param(value: Any, bounds: Any) -> bool:
        if isinstance(value, (int, np.integer)):
            return True
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            lo, hi = bounds[0], bounds[1]
            if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
                return True
        return False

    @staticmethod
    def _mutate_single_param(
        key: str,
        current: Any,
        bounds: Any,
        rng: np.random.Generator,
    ) -> Any:
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
            return current
        low, high = bounds[0], bounds[1]
        if low == high:
            return current

        if ModelTrainer._is_int_param(current, bounds):
            lo_i = int(low)
            hi_i = int(high)
            center = int(round(float(current)))
            width = max(1, int(round((hi_i - lo_i) * 0.25)))
            lo_local = max(lo_i, center - width)
            hi_local = min(hi_i, center + width)
            if lo_local > hi_local:
                lo_local, hi_local = lo_i, hi_i
            return int(rng.integers(lo_local, hi_local + 1))

        lo_f = float(low)
        hi_f = float(high)
        center = float(current)
        width = (hi_f - lo_f) * 0.25
        lo_local = max(lo_f, center - width)
        hi_local = min(hi_f, center + width)
        if lo_local >= hi_local:
            lo_local, hi_local = lo_f, hi_f
        return float(rng.uniform(lo_local, hi_local))

    def _evolutionary_refine(
        self,
        *,
        name: str,
        best_params: Dict[str, Any],
        best_score: float,
        score_fn,
        hpo_space: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        gen_cfg = self.optim_cfg.get("genetic", {}) or {}
        enabled = bool(cfg.get("hpo_genetic_enabled", gen_cfg.get("enabled", False)))
        if not enabled or not best_params:
            return best_params, best_score

        pop_size = int(cfg.get("hpo_population_size", gen_cfg.get("population_size", 8)))
        generations = int(cfg.get("hpo_generations", gen_cfg.get("generations", 2)))
        mutation_rate = float(cfg.get("hpo_mutation_rate", gen_cfg.get("mutation_rate", 0.15)))

        # Keep optimization pragmatic; avoid runaway search budgets.
        pop_size = max(3, min(pop_size, 8))
        generations = max(1, min(generations, 2))
        mutation_rate = min(max(mutation_rate, 0.05), 0.8)

        rng = np.random.default_rng(int(gen_cfg.get("seed", 42)))
        current_best = deepcopy(best_params)
        current_score = float(best_score)
        console_stage(
            f"{self._display_name(name)} evolutionary refine",
            f"generations={generations} population={pop_size} mutation={mutation_rate:.2f}",
            status="info",
        )

        for gen in range(1, generations + 1):
            candidates: List[Dict[str, Any]] = [deepcopy(current_best)]
            for _ in range(pop_size - 1):
                cand = deepcopy(current_best)
                mutated_any = False
                for k, v in list(cand.items()):
                    if k not in hpo_space:
                        continue
                    if rng.uniform() <= mutation_rate:
                        cand[k] = self._mutate_single_param(k, v, hpo_space.get(k), rng)
                        mutated_any = True
                if not mutated_any:
                    # Force at least one mutation to explore locally.
                    keys = [k for k in cand.keys() if k in hpo_space]
                    if keys:
                        k = keys[int(rng.integers(0, len(keys)))]
                        cand[k] = self._mutate_single_param(k, cand[k], hpo_space.get(k), rng)
                candidates.append(cand)

            gen_best_score = current_score
            gen_best_params = deepcopy(current_best)
            for cand in candidates:
                score = float(score_fn(cand))
                if score > gen_best_score:
                    gen_best_score = score
                    gen_best_params = cand

            improved = gen_best_score > current_score
            current_score = gen_best_score
            current_best = gen_best_params
            console_stage(
                f"{self._display_name(name)} evolution g{gen}",
                f"best_score={current_score:.4f} improved={improved}",
                status="ok" if improved else "info",
            )

        return current_best, current_score

    @staticmethod
    def _resolve_classifier_algo(algo: str) -> str:
        if algo == "lightgbm" and lgb is None:
            fallback = "xgboost" if xgb is not None else "logistic"
            LOG.warning("[ModelTrainer] lightgbm unavailable; falling back to %s.", fallback)
            return fallback
        if algo == "xgboost" and xgb is None:
            LOG.warning("[ModelTrainer] xgboost unavailable; falling back to logistic.")
            return "logistic"
        return algo

    @staticmethod
    def _resolve_quantile_algo() -> str:
        if lgb is not None:
            return "lightgbm"
        if xgb is not None:
            LOG.warning("[ModelTrainer] lightgbm unavailable for quantiles; falling back to xgboost quantile regressor.")
            return "xgboost"
        LOG.warning("[ModelTrainer] lightgbm/xgboost unavailable for quantiles; falling back to GradientBoostingRegressor.")
        return "gbr"

    @staticmethod
    def _make_tscv(n_rows: int, requested_splits: int) -> Optional[TimeSeriesSplit]:
        if n_rows < 3:
            return None
        n_splits = max(2, min(int(requested_splits), n_rows - 1))
        if n_splits >= n_rows:
            return None
        return TimeSeriesSplit(n_splits=n_splits)

    @staticmethod
    def _positive_class_proba(model: Any, X_in) -> np.ndarray:
        proba = model.predict_proba(X_in)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.shape[1] == 1:
            classes_attr = getattr(model, "classes_", None)
            classes = list(classes_attr) if classes_attr is not None else []
            if classes == [1]:
                return np.ones(arr.shape[0], dtype=float)
            return np.zeros(arr.shape[0], dtype=float)
        classes_attr = getattr(model, "classes_", None)
        classes = list(classes_attr) if classes_attr is not None else []
        if 1 in classes:
            return arr[:, classes.index(1)]
        return arr[:, min(1, arr.shape[1] - 1)]

    # ------------------------------------------------------------------
    # Utility: class weights and single-class safety
    # ------------------------------------------------------------------
    def _make_class_weight(self, y: np.ndarray, cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[float]]:
        """Compute dynamic class_weight and scale_pos_weight for imbalance-aware training."""
        series = pd.Series(y)
        counts = series.value_counts()
        if counts.empty:
            return None, None
        n_pos = float(counts.get(1, 0.0))
        n_neg = float(counts.get(0, 0.0))
        if n_pos == 0 or n_neg == 0:
            return None, None

        scale_pos = (n_neg / n_pos) if n_pos > 0 else None

        # Explicit override from YAML wins.
        # Supported values:
        #   - "balanced"
        #   - dict mapping class->weight
        #   - "none" / false / null (disable weighting)
        cw_cfg = cfg.get("class_weight")
        if isinstance(cw_cfg, str):
            low = cw_cfg.strip().lower()
            if low in {"none", "off", "false", "null"}:
                return None, scale_pos
            if low == "balanced":
                return "balanced", scale_pos
        elif isinstance(cw_cfg, dict):
            return cw_cfg, scale_pos
        elif cw_cfg is False:
            return None, scale_pos

        # Auto mode: only apply balancing when class skew is meaningfully imbalanced.
        ratio = max(n_pos, n_neg) / max(1.0, min(n_pos, n_neg))
        ratio_threshold = float(cfg.get("imbalance_ratio_threshold", 1.5))
        if ratio < ratio_threshold:
            return None, scale_pos

        total = n_pos + n_neg
        cw = {
            0: total / (2.0 * n_neg),
            1: total / (2.0 * n_pos),
        }
        return cw, scale_pos

    def _candidate_algorithms(self, cfg: Dict[str, Any], default_algo: str = "logistic") -> List[str]:
        base_algo_raw = str(cfg.get("algorithm", default_algo)).lower()
        algos = [self._resolve_classifier_algo(base_algo_raw)]
        if bool(cfg.get("challenger_enabled", False)):
            for raw in cfg.get("challenger_algorithms", []) or []:
                cand = self._resolve_classifier_algo(str(raw).lower())
                if cand not in algos:
                    algos.append(cand)
        return algos

    def _train_with_challengers(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        cfg: Dict[str, Any],
        name: str,
        num_cols: List[str],
        cat_cols: List[str],
        default_algo: str = "logistic",
    ) -> Tuple[Any, Dict[str, Any]]:
        algos = self._candidate_algorithms(cfg, default_algo=default_algo)
        if len(algos) == 1:
            single_cfg = deepcopy(cfg)
            single_cfg["algorithm"] = algos[0]
            model, metrics = self._train_classifier(X_df, y, single_cfg, name=name, num_cols=num_cols, cat_cols=cat_cols)
            metrics = metrics or {}
            metrics["selected_algorithm"] = algos[0]
            metrics["challenger_scores"] = {algos[0]: metrics.get("cv_score")}
            return model, metrics

        challenger_trials = int(cfg.get("challenger_hpo_trials", max(1, int(cfg.get("hpo_trials", 20)) // 2)))
        leaderboard: Dict[str, float] = {}
        best_algo = algos[0]
        best_score = -np.inf
        best_model = None
        best_metrics: Dict[str, Any] = {}

        console_stage(
            f"{self._display_name(name)} challenger run",
            f"algorithms={', '.join(algos)}",
            status="info",
        )

        for idx, algo in enumerate(algos):
            cfg_algo = deepcopy(cfg)
            cfg_algo["algorithm"] = algo
            if idx > 0:
                cfg_algo["hpo_trials"] = challenger_trials
            model, metrics = self._train_classifier(
                X_df,
                y,
                cfg_algo,
                name=f"{name}_{algo}",
                num_cols=num_cols,
                cat_cols=cat_cols,
            )
            score = metrics.get("cv_score")
            score = float(score) if score is not None else float("-inf")
            leaderboard[algo] = score
            if score > best_score:
                best_score = score
                best_algo = algo
                best_model = model
                best_metrics = metrics

        console_stage(
            f"{self._display_name(name)} champion selected",
            f"algo={best_algo} cv_score={best_score:.4f}",
            status="ok",
        )

        if best_model is None:
            # Defensive fallback, should never happen.
            fallback_cfg = deepcopy(cfg)
            fallback_cfg["algorithm"] = algos[0]
            best_model, best_metrics = self._train_classifier(
                X_df, y, fallback_cfg, name=name, num_cols=num_cols, cat_cols=cat_cols
            )
            best_algo = algos[0]

        best_metrics = best_metrics or {}
        best_metrics["selected_algorithm"] = best_algo
        best_metrics["challenger_scores"] = leaderboard
        return best_model, best_metrics

    # ------------------------------------------------------------------
    # Public entry: train all models for an asset
    # ------------------------------------------------------------------
    def train_asset(self, df: pd.DataFrame, asset: str) -> str:
        return self.train_asset_bundle(df, asset)["version"]

    def train_asset_bundle(
        self,
        df: pd.DataFrame,
        asset: str,
        selected_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        requested_models = self._normalize_requested_models(selected_models)
        LOG.info(
            "[ModelTrainer] Training model bundle for asset=%s requested=%s",
            asset,
            requested_models,
        )
        t0 = time.perf_counter()

        targets = {
            "liq_flow": df.get("label_liq_flow", pd.Series(0, index=df.index)).astype(int).values,
            "bos_cont": df.get("label_bos_cont", pd.Series(0, index=df.index)).astype(int).values,
            "flow_1h": df.get("label_flow_1h", pd.Series(0, index=df.index)).astype(int).values,
            "momo": df.get("label_momo", pd.Series(0, index=df.index)).astype(int).values,
            "eop": df.get("label_eop", pd.Series(0, index=df.index)).astype(int).values,
            "edp": df.get("label_edp", pd.Series(0, index=df.index)).astype(int).values,
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

        meta_cfg = self.model_cfg.get("meta_model", {})
        conf_cfg = self.model_cfg.get("confluence_model", {})
        required_specialists: List[str] = []
        for name in requested_models:
            if name in self.SPECIALIST_MODELS:
                required_specialists.append(name)
        if "meta_model" in requested_models:
            required_specialists.extend(meta_cfg.get("specialist_inputs", self.SPECIALIST_MODELS))
        if "confluence_model" in requested_models:
            required_specialists.extend(conf_cfg.get("specialist_inputs", self.SPECIALIST_MODELS))
        required_specialists = list(dict.fromkeys(required_specialists))

        planned_steps = list(required_specialists)
        if "meta_model" in requested_models:
            planned_steps.append("meta_model")
        if "confluence_model" in requested_models:
            planned_steps.append("confluence_model")
        if "hazard" in requested_models:
            planned_steps.append("hazard")
        if "quantile" in requested_models:
            planned_steps.append("quantile")
        total_steps = len(planned_steps)
        completed_steps = 0
        console_kv(
            "Training Bundle Plan",
            {
                "asset": asset,
                "requested_models": ", ".join(requested_models),
                "planned_steps": fmt_num(total_steps),
                "order": " -> ".join(planned_steps) if planned_steps else "-",
                "heartbeat_seconds": self.heartbeat_seconds,
            },
            style="magenta",
        )

        # Specialist models ------------------------------------------------
        specialists = {}
        specialist_metrics = {}
        for key in required_specialists:
            cfg = self.model_cfg[key]
            X_sel, cols_sel, num_cols, cat_cols = self._prepare_features(
                df,
                feature_spec=cfg.get("features", {}),
                task_name=key,
            )
            LOG.info(f"[ModelTrainer] Specialist {key} | features={len(cols_sel)}")
            t_spec = time.perf_counter()
            model, metrics = self._train_classifier(
                X_sel, targets[key], cfg, name=f"{asset}_{key}", num_cols=num_cols, cat_cols=cat_cols
            )
            model = self._calibrate(model, X_sel, targets[key], cfg.get("calibrator"))
            selected_cols = metrics.get("selected_feature_cols", cols_sel) if isinstance(metrics, dict) else cols_sel
            specialists[key] = {"model": model, "feature_cols": selected_cols}
            specialist_metrics[key] = metrics
            LOG.info(f"[ModelTrainer] Specialist {key} done in {time.perf_counter() - t_spec:.2f}s")
            completed_steps += 1
            self._log_bundle_progress(
                asset=asset,
                completed=completed_steps,
                total=total_steps,
                phase=f"specialist:{key}",
                started_at=t0,
            )

        meta_model = meta_meta = None
        if "meta_model" in requested_models:
            LOG.info("[ModelTrainer] Training meta model (stacking)")
            meta_model, meta_meta = self._train_stack(df, specialists, meta_cfg, target_key="label_liq_flow")
            LOG.info("[ModelTrainer] Meta model done")
            completed_steps += 1
            self._log_bundle_progress(
                asset=asset,
                completed=completed_steps,
                total=total_steps,
                phase="meta_model",
                started_at=t0,
            )

        conf_model = conf_meta = None
        if "confluence_model" in requested_models:
            LOG.info("[ModelTrainer] Training confluence model")
            conf_model, conf_meta = self._train_stack(df, specialists, conf_cfg, target_key="label_liq_flow")
            LOG.info("[ModelTrainer] Confluence model done")
            completed_steps += 1
            self._log_bundle_progress(
                asset=asset,
                completed=completed_steps,
                total=total_steps,
                phase="confluence_model",
                started_at=t0,
            )

        hazard_models = haz_conf = None
        if "hazard" in requested_models:
            haz_cfg = self.model_cfg.get("hazard", {})
            LOG.info("[ModelTrainer] Training hazard models")
            hazard_models, haz_conf = self._train_hazard(df, haz_cfg, haz_event, haz_time)
            LOG.info(f"[ModelTrainer] Hazard models done ({len(hazard_models)} bins)")
            completed_steps += 1
            self._log_bundle_progress(
                asset=asset,
                completed=completed_steps,
                total=total_steps,
                phase="hazard",
                started_at=t0,
            )

        quant_models = quant_conf = None
        if "quantile" in requested_models:
            q_cfg = self.model_cfg.get("quantile_forecaster", {})
            LOG.info("[ModelTrainer] Training quantile forecaster")
            quant_models, quant_conf = self._train_quantile(df, q_cfg, prices)
            LOG.info("[ModelTrainer] Quantile forecaster done")
            completed_steps += 1
            self._log_bundle_progress(
                asset=asset,
                completed=completed_steps,
                total=total_steps,
                phase="quantile",
                started_at=t0,
            )

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

        if meta_model is not None and meta_meta is not None:
            self.registry.save_model(
                model_name=f"{asset}_meta",
                version=version,
                clf=meta_model,
                cal=None,
                config=meta_meta,
            )
            self.registry.save_model(
                model_name="meta_model",
                version=version,
                clf=meta_model,
                cal=None,
                config=meta_meta,
            )
        if conf_model is not None and conf_meta is not None:
            self.registry.save_model(
                model_name=f"{asset}_confluence",
                version=version,
                clf=conf_model,
                cal=None,
                config=conf_meta,
            )
            self.registry.save_model(
                model_name="confluence_model",
                version=version,
                clf=conf_model,
                cal=None,
                config=conf_meta,
            )

        if hazard_models is not None and haz_conf is not None:
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

        if quant_models is not None and quant_conf is not None:
            self.registry.save_model(
                model_name=f"{asset}_quantile",
                version=version,
                clf=quant_models,
                cal=None,
                config=quant_conf,
            )
            self.registry.save_model(
                model_name="quantile",
                version=version,
                clf=quant_models,
                cal=None,
                config=quant_conf,
            )

        persisted = list(dict.fromkeys(
            list(specialists.keys())
            + (["meta_model"] if meta_model is not None else [])
            + (["confluence_model"] if conf_model is not None else [])
            + (["hazard"] if hazard_models is not None else [])
            + (["quantile"] if quant_models is not None else [])
        ))
        dependency_models = [m for m in specialists.keys() if m not in requested_models]
        LOG.info(
            "[ModelTrainer] Completed asset=%s version=%s models=%s elapsed=%.2fs",
            asset,
            version,
            persisted,
            time.perf_counter() - t0,
        )
        return {
            "asset": asset,
            "version": version,
            "requested_models": requested_models,
            "dependency_models": dependency_models,
            "trained_models": persisted,
            "specialist_metrics": specialist_metrics,
        }

    @classmethod
    def _normalize_requested_models(cls, selected_models: Optional[List[str]]) -> List[str]:
        if not selected_models:
            return list(cls.TRAINABLE_MODELS)
        normalized: List[str] = []
        for raw in selected_models:
            key = str(raw).strip()
            if not key:
                continue
            key = cls.MODEL_NAME_ALIASES.get(key, key)
            if key not in cls.TRAINABLE_MODELS:
                raise ValueError(f"Unknown trainable model: {raw}")
            if key not in normalized:
                normalized.append(key)
        if not normalized:
            raise ValueError("At least one trainable model must be selected.")
        return normalized

    # ------------------------------------------------------------------
    # Feature selection helpers
    # ------------------------------------------------------------------
    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_spec: Optional[Dict[str, Any]],
        task_name: Optional[str] = None,
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
            "label_flow_1h",
            "label_momo",
            "label_eop",
            "label_edp",
            "hazard_event",
            "hazard_time",
        }
        all_feature_cols = [
            c for c in df.columns
            if c not in drop_cols and not self._is_model_output_column(c)
        ]

        feature_spec = feature_spec or {}
        groups = feature_spec.get("group", []) if isinstance(feature_spec, dict) else []
        contains_any = feature_spec.get("contains_any", []) if isinstance(feature_spec, dict) else []
        prefixes_any = feature_spec.get("prefixes_any", []) if isinstance(feature_spec, dict) else []
        exact = feature_spec.get("exact", []) if isinstance(feature_spec, dict) else []
        exclude_contains = feature_spec.get("exclude_contains", []) if isinstance(feature_spec, dict) else []
        exclude_prefixes = feature_spec.get("exclude_prefixes", []) if isinstance(feature_spec, dict) else []
        exclude_exact = set(feature_spec.get("exclude_exact", [])) if isinstance(feature_spec, dict) else set()

        if not groups and not contains_any and not prefixes_any and not exact:
            use_cols = all_feature_cols
        else:
            prefixes = {
                "smc": (
                    "swing_",
                    "bos_",
                    "choch_",
                    "bias",
                    "structure_bias",
                    "structural_bias",
                    "fvg_",
                    "sweep_",
                    "demand_",
                    "supply_",
                    "compression_",
                    "pd_",
                    "zone_",
                    "fresh_retest_",
                    "retest_",
                    "displacement_",
                ),
                "ema": ("ema_", "dist_to_ema", "band_", "sma_", "hma_", "ema_rel"),
                "volatility": ("atr", "vol_", "rng_", "range_", "rstd", "ret_", "realized_vol", "volatility_", "vol_z"),
                "liquidity": ("liq_", "wick_", "eql_", "spread_", "dollar_volume", "absorption_"),
                "regime": ("p_regime_", "regime_", "vol_pct", "trend_persist", "compression_12h", "toxicity_"),
                "session": ("session_", "is_ldn", "is_ny", "is_asia", "is_overlap"),
            }
            use_cols = []
            for col in all_feature_cols:
                selected = False
                for g in groups:
                    if any(col.startswith(p) for p in prefixes.get(g, ())):
                        selected = True
                        break
                if not selected and any(tok in col for tok in contains_any):
                    selected = True
                if not selected and any(col.startswith(tok) for tok in prefixes_any):
                    selected = True
                if not selected and col in exact:
                    selected = True
                if selected:
                    use_cols.append(col)
            if not use_cols:
                use_cols = all_feature_cols

        filtered_cols = []
        for col in use_cols:
            if col in exclude_exact:
                continue
            if any(tok in col for tok in exclude_contains):
                continue
            if any(col.startswith(tok) for tok in exclude_prefixes):
                continue
            filtered_cols.append(col)
        use_cols = filtered_cols or use_cols
        use_cols = self._apply_timeframe_scope(use_cols, feature_spec or {}, task_name)

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

        X_df = self._apply_feature_hygiene(X_df, task_name=task_name)

        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]
        return X_df, list(X_df.columns), num_cols, cat_cols

    @classmethod
    def _is_model_output_column(cls, col: str) -> bool:
        if col.startswith("label_"):
            return True
        if col.startswith("prob_"):
            return True
        if col.startswith("p_") and col[2:] in cls.PREDICTION_OUTPUTS:
            return True
        if col in {"hazard", "hazard_score", "prob_meta", "prob_confluence", "confluence_score"}:
            return True
        if col == "quantiles" or col.startswith("quantiles_"):
            return True
        if col.startswith("hazard_curve"):
            return True
        if re.match(r"^q(_?\d|[0-9])", col):
            return True
        return False

    def _apply_feature_hygiene(self, X_df: pd.DataFrame, task_name: Optional[str] = None) -> pd.DataFrame:
        cfg = self.feature_sel_cfg if isinstance(self.feature_sel_cfg, dict) else {}
        if not bool(cfg.get("enabled", True)):
            return X_df

        min_features = max(int(cfg.get("min_features", 24)), 1)
        dropped: List[str] = []
        out = X_df.copy()

        max_missing_ratio = float(cfg.get("max_missing_ratio", 0.35))
        if 0.0 <= max_missing_ratio < 1.0 and len(out.columns) > min_features:
            miss = out.isna().mean()
            miss_drop = [c for c in miss.index if miss[c] > max_missing_ratio]
            room = max(len(out.columns) - min_features, 0)
            miss_drop = miss_drop[:room]
            if miss_drop:
                out = out.drop(columns=miss_drop, errors="ignore")
                dropped.extend(miss_drop)

        if bool(cfg.get("drop_exact_duplicates", True)) and len(out.columns) > min_features:
            sig_to_col: Dict[int, str] = {}
            dup_cols: List[str] = []
            for col in list(out.columns):
                series = out[col]
                try:
                    sig = int(pd.util.hash_pandas_object(series, index=False).sum())
                except Exception:
                    continue
                twin = sig_to_col.get(sig)
                if twin is not None and series.equals(out[twin]):
                    dup_cols.append(col)
                else:
                    sig_to_col[sig] = col
            room = max(len(out.columns) - min_features, 0)
            dup_cols = dup_cols[:room]
            if dup_cols:
                out = out.drop(columns=dup_cols, errors="ignore")
                dropped.extend(dup_cols)

        corr_threshold = float(cfg.get("correlation_threshold", 0.90))
        if 0.0 < corr_threshold < 1.0 and len(out.columns) > min_features:
            num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
            if len(num_cols) >= 2:
                corr = out[num_cols].corr(method="pearson").abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                corr_drop = [col for col in upper.columns if (upper[col] >= corr_threshold).any()]
                room = max(len(out.columns) - min_features, 0)
                corr_drop = corr_drop[:room]
                if corr_drop:
                    out = out.drop(columns=corr_drop, errors="ignore")
                    dropped.extend(corr_drop)

        if dropped:
            task = task_name or "generic"
            LOG.info(
                "[ModelTrainer] Feature hygiene %s dropped=%s remaining=%s",
                task,
                len(dropped),
                len(out.columns),
            )
        return out

    def _apply_mutual_info_filter(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cfg = self.feature_sel_cfg if isinstance(self.feature_sel_cfg, dict) else {}
        report: Dict[str, Any] = {"enabled": False}
        if not bool(cfg.get("enabled", True)):
            return X_df, report
        if not bool(cfg.get("use_mutual_info", True)):
            return X_df, report
        if len(X_df.columns) <= 1:
            return X_df, report
        if pd.Series(y).nunique() < 2:
            return X_df, report

        top_k = int(cfg.get("mutual_info_top_k", 180))
        min_features = max(int(cfg.get("min_features", 24)), 1)
        keep_n = min(max(min_features, top_k), len(X_df.columns))
        if keep_n >= len(X_df.columns):
            return X_df, report

        work = X_df.copy()
        num_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        cat_cols = [c for c in work.columns if c not in num_cols]

        for col in num_cols:
            med = work[col].median()
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(med if pd.notna(med) else 0.0)
        for col in cat_cols:
            vals = work[col].astype(str).fillna("__nan__")
            work[col] = pd.factorize(vals, sort=False)[0]

        discrete_mask = np.array([c in cat_cols for c in work.columns], dtype=bool)
        random_state = int(cfg.get("mutual_info_random_state", 42))
        try:
            scores = mutual_info_classif(
                work.values,
                y,
                discrete_features=discrete_mask,
                random_state=random_state,
            )
        except Exception as exc:
            LOG.warning("[ModelTrainer] mutual_info skipped for %s: %s", model_name, exc)
            return X_df, report

        score_s = pd.Series(scores, index=work.columns).fillna(0.0)
        keep_ranked = list(score_s.sort_values(ascending=False).head(keep_n).index)
        keep_set = set(keep_ranked)
        keep_cols = [c for c in X_df.columns if c in keep_set]
        if len(keep_cols) < min_features:
            return X_df, report

        top_features = score_s.sort_values(ascending=False).head(min(10, len(score_s)))
        report = {
            "enabled": True,
            "method": "mutual_info",
            "selected": int(len(keep_cols)),
            "dropped": int(len(X_df.columns) - len(keep_cols)),
            "top_features": {k: float(v) for k, v in top_features.items()},
        }
        LOG.info(
            "[ModelTrainer] MI filter %s selected=%s/%s",
            model_name,
            len(keep_cols),
            len(X_df.columns),
        )
        return X_df[keep_cols].copy(), report

    @staticmethod
    def _extract_model_importance(model: Any) -> Dict[str, Any]:
        try:
            clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
            pre = model.named_steps.get("pre") if hasattr(model, "named_steps") else None
            if clf is None:
                return {}
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                names = [str(x) for x in pre.get_feature_names_out()]
            else:
                names = []

            scores = None
            if hasattr(clf, "feature_importances_"):
                raw = np.asarray(clf.feature_importances_, dtype=float)
                scores = np.abs(raw)
            elif hasattr(clf, "coef_"):
                coef = np.asarray(clf.coef_, dtype=float)
                scores = np.abs(coef).mean(axis=0)

            if scores is None or scores.size == 0:
                return {}
            if not names or len(names) != len(scores):
                names = [f"f{i}" for i in range(len(scores))]
            ser = pd.Series(scores, index=names).sort_values(ascending=False)
            top = ser.head(20)
            return {"top_features": {k: float(v) for k, v in top.items()}}
        except Exception:
            return {}

    def _permutation_importance_audit(
        self,
        model: Any,
        X_df: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        cfg = self.feature_sel_cfg if isinstance(self.feature_sel_cfg, dict) else {}
        if not bool(cfg.get("enabled", True)):
            return {}
        if not bool(cfg.get("use_permutation_audit", False)):
            return {}
        if len(X_df.columns) <= 1 or len(X_df) < 200:
            return {}

        sample_rows = int(cfg.get("permutation_sample_rows", 6000))
        top_n = int(cfg.get("permutation_top_n", 20))
        repeats = int(cfg.get("permutation_n_repeats", 4))
        random_state = int(cfg.get("mutual_info_random_state", 42))

        if len(X_df) > sample_rows:
            X_eval = X_df.tail(sample_rows).copy()
            y_eval = y[-sample_rows:]
        else:
            X_eval = X_df
            y_eval = y

        try:
            imp = permutation_importance(
                model,
                X_eval,
                y_eval,
                scoring="average_precision",
                n_repeats=max(repeats, 1),
                random_state=random_state,
                n_jobs=1,
            )
        except Exception as exc:
            LOG.warning("[ModelTrainer] permutation_importance skipped: %s", exc)
            return {}

        ser = pd.Series(imp.importances_mean, index=list(X_eval.columns)).sort_values(ascending=False)
        top = ser.head(max(top_n, 1))
        return {"top_features": {k: float(v) for k, v in top.items()}}

    def _feature_dominance_audit(
        self,
        model: Any,
        X_df: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        cfg = self.feature_sel_cfg if isinstance(self.feature_sel_cfg, dict) else {}
        audit_cfg = cfg.get("dominance_audit", {}) if isinstance(cfg.get("dominance_audit", {}), dict) else {}
        if not bool(audit_cfg.get("enabled", False)):
            return {}

        monitor_by_model = audit_cfg.get("monitor_features_by_model", {})
        monitor_features: List[str] = []
        if isinstance(monitor_by_model, dict):
            raw = monitor_by_model.get(model_name, [])
            if isinstance(raw, list):
                monitor_features = [str(x) for x in raw if str(x) in X_df.columns]
        if not monitor_features:
            return {}

        min_rows = max(int(audit_cfg.get("min_rows", 4096)), 256)
        if len(X_df) < min_rows or len(X_df.columns) <= 1:
            return {}

        sample_rows = max(int(audit_cfg.get("sample_rows", 20000)), min_rows)
        random_state = int(audit_cfg.get("random_state", 42))
        min_top_drop = float(audit_cfg.get("min_top_drop", 0.01))
        warn_share = float(audit_cfg.get("dominance_share_warn", 0.55))

        if len(X_df) > sample_rows:
            X_eval = X_df.tail(sample_rows).copy()
            y_eval = y[-sample_rows:]
        else:
            X_eval = X_df.copy()
            y_eval = y

        if pd.Series(y_eval).nunique() < 2:
            return {}

        try:
            p_base = self._positive_class_proba(model, X_eval)
            base_ap = float(average_precision_score(y_eval, p_base))
        except Exception as exc:
            LOG.warning("[ModelTrainer] dominance audit skipped for %s: %s", model_name, exc)
            return {}

        rng = np.random.default_rng(random_state)
        rows: List[Dict[str, Any]] = []
        for idx, feat in enumerate(monitor_features):
            X_perm = X_eval.copy()
            values = X_perm[feat].to_numpy()
            perm = rng.permutation(len(values))
            X_perm[feat] = values[perm]
            try:
                p_perm = self._positive_class_proba(model, X_perm)
                ap_perm = float(average_precision_score(y_eval, p_perm))
            except Exception:
                ap_perm = base_ap
            drop = float(base_ap - ap_perm)
            corr_val: Optional[float] = None
            try:
                feat_num = pd.to_numeric(X_eval[feat], errors="coerce")
                corr_tmp = feat_num.corr(pd.Series(y_eval))
                if pd.notna(corr_tmp):
                    corr_val = float(corr_tmp)
            except Exception:
                corr_val = None
            rows.append(
                {
                    "feature": feat,
                    "ap_drop": drop,
                    "ap_permuted": ap_perm,
                    "label_corr": corr_val,
                    "rank_seed_offset": idx,
                }
            )

        if not rows:
            return {}

        rows_sorted = sorted(rows, key=lambda r: r["ap_drop"], reverse=True)
        positive_sum = float(sum(max(float(r["ap_drop"]), 0.0) for r in rows_sorted))
        top = rows_sorted[0]
        dominance_share = (float(top["ap_drop"]) / positive_sum) if positive_sum > 0 else None
        warn = bool(
            dominance_share is not None
            and float(top["ap_drop"]) >= min_top_drop
            and float(dominance_share) >= warn_share
        )
        if warn:
            LOG.warning(
                "[ModelTrainer] dominance audit %s top_feature=%s ap_drop=%.4f share=%.3f (threshold=%.3f)",
                model_name,
                top["feature"],
                float(top["ap_drop"]),
                float(dominance_share),
                float(warn_share),
            )

        return {
            "enabled": True,
            "model": model_name,
            "rows_evaluated": int(len(X_eval)),
            "base_ap": base_ap,
            "monitored_features": monitor_features,
            "ranked_drops": rows_sorted,
            "top_feature": top["feature"],
            "top_ap_drop": float(top["ap_drop"]),
            "top_dominance_share": float(dominance_share) if dominance_share is not None else None,
            "warn": warn,
            "warn_thresholds": {
                "min_top_drop": float(min_top_drop),
                "dominance_share_warn": float(warn_share),
            },
        }

    @staticmethod
    def _infer_timeframe(col: str) -> Optional[str]:
        match = re.search(r"_(15m|1h|6h|12h)$", col)
        return match.group(1) if match else None

    @staticmethod
    def _base_name(col: str) -> str:
        return re.sub(r"_(15m|1h|6h|12h)$", "", col)

    def _apply_timeframe_scope(
        self,
        cols: List[str],
        feature_spec: Dict[str, Any],
        task_name: Optional[str],
    ) -> List[str]:
        scoped = feature_spec.copy()
        if task_name == "flow_1h" and not any(
            k in scoped for k in ("allowed_timeframes", "allow_unsuffixed", "timeframe_prefixes")
        ):
            scoped.update(
                {
                    "allowed_timeframes": ["1h", "6h", "12h"],
                    "allow_unsuffixed": False,
                    "unsuffixed_exact": [
                        "session",
                        "session_weight",
                        "regime_state",
                        "regime_state_id",
                        "atr",
                        "atr_15m",
                        "realized_vol",
                        "vol_zscore",
                        "range_pct",
                        "vol_pct",
                        "trend_persist",
                    ],
                    "unsuffixed_prefixes": ["p_regime_"],
                    "timeframe_prefixes": {
                        "6h": ["zone_", "compression_", "bias", "structural_bias", "demand_", "supply_"],
                        "12h": ["regime_", "p_regime_", "toxicity_", "compression_", "vol_pct"],
                    },
                }
            )

        allowed_timeframes = set(scoped.get("allowed_timeframes", []) or [])
        exclude_timeframes = set(scoped.get("exclude_timeframes", []) or [])
        allow_unsuffixed = bool(scoped.get("allow_unsuffixed", True))
        unsuffixed_exact = set(scoped.get("unsuffixed_exact", []) or [])
        unsuffixed_prefixes = tuple(scoped.get("unsuffixed_prefixes", []) or [])
        tf_prefixes = scoped.get("timeframe_prefixes", {}) or {}

        if not (allowed_timeframes or exclude_timeframes or not allow_unsuffixed or unsuffixed_exact or unsuffixed_prefixes or tf_prefixes):
            return cols

        selected: List[str] = []
        for col in cols:
            tf = self._infer_timeframe(col)
            base = self._base_name(col)
            if tf is None:
                if allow_unsuffixed or col in unsuffixed_exact or any(col.startswith(p) for p in unsuffixed_prefixes):
                    selected.append(col)
                continue
            if allowed_timeframes and tf not in allowed_timeframes:
                continue
            if tf in exclude_timeframes:
                continue
            allowed_prefixes = tuple(tf_prefixes.get(tf, []) or [])
            if allowed_prefixes and not any(base.startswith(prefix) for prefix in allowed_prefixes):
                continue
            selected.append(col)

        return selected or cols

    def _build_preprocessor(self, num_cols: List[str], cat_cols: List[str], algo: Optional[str] = None) -> ColumnTransformer:
        algo = (algo or "").lower()
        num_imputer = str(self.preproc_cfg.get("num_imputer", "median")).lower()
        cat_imputer = str(self.preproc_cfg.get("cat_imputer", "most_frequent")).lower()
        scaler_mode = str(self.preproc_cfg.get("scaler", "standard")).lower()
        scale_for_tree = bool(self.preproc_cfg.get("scale_for_tree_models", True))
        clip_enabled = bool(self.preproc_cfg.get("outlier_clip", True))
        clip_q = self.preproc_cfg.get("clip_quantiles", [0.005, 0.995])
        try:
            lower_q = float(clip_q[0])
            upper_q = float(clip_q[1])
        except Exception:
            lower_q, upper_q = 0.005, 0.995
        lower_q = min(max(lower_q, 0.0), 0.49)
        upper_q = max(min(upper_q, 1.0), 0.51)

        num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_imputer))]
        if clip_enabled:
            num_steps.append(("clip", QuantileClipper(lower_q=lower_q, upper_q=upper_q)))

        is_tree = algo in {"lightgbm", "xgboost", "gbr"}
        if scaler_mode != "none" and (scale_for_tree or not is_tree):
            if scaler_mode == "robust":
                num_steps.append(("scaler", RobustScaler()))
            else:
                num_steps.append(("scaler", StandardScaler()))

        num_pipe = Pipeline(steps=num_steps)
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=cat_imputer)),
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
        algo = self._resolve_classifier_algo(cfg.get("algorithm", "lightgbm").lower())
        n_trials = int(cfg.get("hpo_trials", 30))
        cv_splits = int(cfg.get("cv_splits", 4))
        hpo_space = cfg.get("hpo_space", {})

        X_df, mi_report = self._apply_mutual_info_filter(X_df, y, name)
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]

        # handle imbalance & single-class upfront
        cw, scale_pos = self._make_class_weight(y, cfg)
        class_counts = pd.Series(y).value_counts().to_dict()
        n_pos = float(class_counts.get(1, 0.0))
        n_neg = float(class_counts.get(0, 0.0))
        imbalance_ratio = (max(n_pos, n_neg) / max(1.0, min(n_pos, n_neg))) if (n_pos > 0 and n_neg > 0) else None
        if cw:
            LOG.info(
                f"[ModelTrainer] {name} class_weight={cw} scale_pos_weight={scale_pos} "
                f"imbalance_ratio={imbalance_ratio} counts={class_counts}"
            )
        else:
            LOG.info(
                f"[ModelTrainer] {name} class counts={class_counts} "
                f"(no weighting applied, imbalance_ratio={imbalance_ratio})"
            )
        self._log_training_header(
            name=name,
            algo=algo,
            rows=len(X_df),
            features=len(X_df.columns),
            class_counts=class_counts,
            trials=n_trials,
        )
        # single-class guard: fall back to a constant predictor so pipeline doesn't explode
        if pd.Series(y).nunique() < 2:
            LOG.warning(f"[ModelTrainer] {name} target has single class; training DummyClassifier.")
            console_stage(
                f"{self._display_name(name)} single-class target",
                f"class={next(iter(class_counts.keys())) if class_counts else 'n/a'} -> DummyClassifier",
                status="warn",
            )
            pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
            dummy = DummyClassifier(strategy="most_frequent")
            model = Pipeline([("pre", pre), ("clf", dummy)])
            with self._suppress_low_signal_warnings():
                model.fit(X_df, y)
            metrics = {"cv_score": None, "best_params": {}, "hpo_trials": 0, "class_counts": class_counts}
            metrics["selected_feature_cols"] = list(X_df.columns)
            metrics["feature_selection"] = mi_report
            metrics["model_importance"] = self._extract_model_importance(model)
            metrics["feature_dominance_audit"] = {}
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

        default_params = self._default_params_for_algo(algo, hpo_space)

        def param_space(trial) -> Dict[str, Any]:
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

        def ts_score(
            params: Dict[str, Any],
            *,
            trial_label: Optional[str] = None,
            trial_started_at: Optional[float] = None,
        ) -> float:
            base_model = build_model(params)
            pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
            model = Pipeline([("pre", pre), ("clf", base_model)])
            tscv = self._make_tscv(len(X_df), cv_splits)
            if tscv is None:
                with self._suppress_low_signal_warnings():
                    model.fit(X_df, y)
                    prob = self._positive_class_proba(model, X_df)
                    try:
                        return float(average_precision_score(y, prob))
                    except Exception:
                        return 0.5
            scores = []
            splits = list(tscv.split(X_df))
            last_beat = trial_started_at or time.perf_counter()
            for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
                Xt, Xv = X_df.iloc[tr_idx], X_df.iloc[va_idx]
                yt, yv = y[tr_idx], y[va_idx]
                with self._suppress_low_signal_warnings():
                    model.fit(Xt, yt)
                    prob = self._positive_class_proba(model, Xv)
                    pr = average_precision_score(yv, prob)
                    try:
                        auc = roc_auc_score(yv, prob)
                    except Exception:
                        auc = 0.5
                scores.append(pr if not np.isnan(pr) else auc)
                now = time.perf_counter()
                if trial_label and (now - last_beat) >= self.heartbeat_seconds:
                    elapsed = now - (trial_started_at or now)
                    fold_done = fold_idx
                    fold_total = max(len(splits), 1)
                    rate = fold_done / max(elapsed, 1e-6)
                    eta = (fold_total - fold_done) / max(rate, 1e-6)
                    partial = float(np.mean(scores)) if scores else 0.0
                    console_stage(
                        "Training heartbeat",
                        (
                            f"{self._display_name(name)} {trial_label} "
                            f"fold={fold_done}/{fold_total} "
                            f"partial_score={partial:.4f} "
                            f"elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
                        ),
                        status="info",
                    )
                    last_beat = now
            return float(np.mean(scores))

        def objective(trial):
            params = param_space(trial)
            params = {k: v for k, v in params.items() if v is not None}
            return -ts_score(
                params,
                trial_label=f"trial {trial.number + 1}/{n_trials}",
                trial_started_at=time.perf_counter(),
            )

        def _cb(study, trial):
            if trial.value is None:
                return
            should_log = (
                trial.number == 0
                or (trial.number + 1) == n_trials
                or ((trial.number + 1) % max(1, min(5, n_trials // 4 or 1)) == 0)
            )
            if should_log:
                self._log_hpo_progress(
                    name=name,
                    trial_number=trial.number + 1,
                    total_trials=n_trials,
                    trial_value=float(trial.value),
                    best_value=float(study.best_value),
                    started_at=hpo_t0,
                )

        if optuna is None or n_trials <= 0:
            LOG.warning("[ModelTrainer] Optuna unavailable for %s; using default params=%s", name, default_params)
            best_params = {k: v for k, v in default_params.items() if v is not None}
            best_cv = ts_score(best_params, trial_label="default", trial_started_at=time.perf_counter())
        else:
            study = self._make_study(direction="minimize", cfg=cfg)
            if study is None:
                best_params = {k: v for k, v in default_params.items() if v is not None}
                best_cv = ts_score(best_params, trial_label="default", trial_started_at=time.perf_counter())
                pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
                final_model = Pipeline([("pre", pre), ("clf", build_model(best_params))])
                with self._suppress_low_signal_warnings():
                    final_model.fit(X_df, y)
                metrics = {
                    "cv_score": best_cv,
                    "best_params": best_params,
                    "hpo_trials": 0,
                    "class_counts": class_counts,
                    "selected_feature_cols": list(X_df.columns),
                    "feature_selection": mi_report,
                    "model_importance": self._extract_model_importance(final_model),
                    "permutation_importance": self._permutation_importance_audit(final_model, X_df, y),
                    "feature_dominance_audit": self._feature_dominance_audit(final_model, X_df, y, name),
                }
                return final_model, metrics
            with self._suppress_low_signal_warnings():
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_cb])
            best_params = study.best_params
            best_cv = -study.best_value
            LOG.info(
                f"[ModelTrainer] Best params {name}: {best_params} | cv_score={best_cv:.4f} "
                f"HPO elapsed={time.perf_counter() - hpo_t0:.1f}s"
            )
            self._log_hpo_summary(name, best_cv, best_params, hpo_t0)
            best_params = {k: v for k, v in best_params.items() if v is not None}

        best_params, best_cv = self._evolutionary_refine(
            name=name,
            best_params=best_params,
            best_score=best_cv,
            score_fn=lambda p: ts_score(p, trial_label="evolution", trial_started_at=time.perf_counter()),
            hpo_space=hpo_space,
            cfg=cfg,
        )

        pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
        final_model = Pipeline([("pre", pre), ("clf", build_model(best_params))])
        with self._suppress_low_signal_warnings():
            final_model.fit(X_df, y)
        importance_report = self._extract_model_importance(final_model)
        perm_report = self._permutation_importance_audit(final_model, X_df, y)
        metrics = {
            "cv_score": best_cv,
            "best_params": best_params,
            "hpo_trials": n_trials,
            # record label balance for downstream monitoring
            "class_counts": class_counts,
            "selected_feature_cols": list(X_df.columns),
            "feature_selection": mi_report,
            "model_importance": importance_report,
            "permutation_importance": perm_report,
            "feature_dominance_audit": self._feature_dominance_audit(final_model, X_df, y, name),
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
            with self._suppress_low_signal_warnings():
                calib.fit(X_df, y)
            return calib

        p_raw = self._positive_class_proba(model, X_df)
        # If probabilities are degenerate, skip calibration
        if np.unique(p_raw).size < 2:
            LOG.warning("[ModelTrainer] Skipping calibration: predicted probabilities are constant.")
            cal = None
        else:
            cal = EmpiricalCalibrator(method="auto").calibrate(p_raw, y)

        return CalibratedProbabilityModel(model, cal)

    # ------------------------------------------------------------------
    # Stacking (meta / confluence)
    # ------------------------------------------------------------------
    def _train_stack(self, df: pd.DataFrame, specialists: Dict[str, Dict[str, Any]], cfg: Dict[str, Any], target_key: str):
        inputs = cfg.get("specialist_inputs", list(specialists.keys()))
        cv_splits = int(cfg.get("cv_splits", 3))
        y_meta = df[target_key].astype(int).values if target_key in df else df["label_liq_flow"].astype(int).values

        self._log_training_header(
            name=f"{target_key}_stack",
            algo=f"{cfg.get('algorithm', 'logistic')}_stack",
            rows=len(df),
            features=len(inputs),
            class_counts=pd.Series(y_meta).value_counts().to_dict(),
            trials=int(cfg.get("hpo_trials", 10)),
        )

        base_cols = []
        for key in inputs:
            model = specialists[key]["model"]
            cols = specialists[key]["feature_cols"]
            base_cols.append((key, model, cols))

        X_meta_parts = []
        tscv = self._make_tscv(len(df), cv_splits)
        if tscv is None:
            full_idx = np.arange(len(df))
            fold_preds = []
            for _, model, cols in base_cols:
                X_fold = df.iloc[full_idx][cols]
                prob = self._positive_class_proba(model, X_fold)
                fold_preds.append(prob)
            X_meta_parts.append((full_idx, np.vstack(fold_preds).T))
        else:
            for _, val_idx in tscv.split(df):
                fold_preds = []
                for _, model, cols in base_cols:
                    X_fold = df.iloc[val_idx][cols]
                    prob = self._positive_class_proba(model, X_fold)
                    fold_preds.append(prob)
                X_fold_mat = np.vstack(fold_preds).T
                X_meta_parts.append((val_idx, X_fold_mat))

        meta_columns = [f"p_{key}" for key, _, _ in base_cols]
        meta_mat = np.zeros((len(df), len(base_cols)))
        filled_mask = np.zeros(len(df), dtype=bool)
        for val_idx, mat in X_meta_parts:
            meta_mat[val_idx, :] = mat
            filled_mask[val_idx] = True

        valid_mask = filled_mask.copy()
        if not valid_mask.any():
            valid_mask = np.ones(len(df), dtype=bool)

        X_meta_df = pd.DataFrame(meta_mat, columns=meta_columns)
        X_train_df = X_meta_df.loc[valid_mask].reset_index(drop=True)
        y_train = y_meta[valid_mask]

        stack_model, stack_metrics = self._train_with_challengers(
            X_df=X_train_df,
            y=y_train,
            cfg=cfg,
            name=f"{target_key}_stack",
            num_cols=list(X_train_df.columns),
            cat_cols=[],
            default_algo="logistic",
        )

        LOG.info(
            "[ModelTrainer] Stacking model trained algo=%s metrics=%s",
            stack_metrics.get("selected_algorithm"),
            {k: v for k, v in stack_metrics.items() if k in {"cv_score", "best_params", "selected_algorithm"}},
        )
        return stack_model, {
            "stack_inputs": inputs,
            "meta_feature_cols": meta_columns,
            "selected_algorithm": stack_metrics.get("selected_algorithm"),
            "challenger_scores": stack_metrics.get("challenger_scores", {}),
            "training_metrics": {k: v for k, v in stack_metrics.items() if k not in {"challenger_scores"}},
        }

    # ------------------------------------------------------------------
    # Hazard model (per-bin logistic)
    # ------------------------------------------------------------------
    def _train_hazard(self, df: pd.DataFrame, cfg: Dict[str, Any], event: np.ndarray, tte: np.ndarray):
        horizon = int(cfg.get("horizon_bars", 48))
        hpo_trials = int(cfg.get("hpo", {}).get("trials_per_bin", 5))
        cv_splits = int(cfg.get("cv_splits", 3))
        hpo_space = cfg.get("hpo_space", {"C": [0.01, 10.0]})
        base_algo = self._resolve_classifier_algo(str(cfg.get("algorithm", "logistic")).lower())
        selected_algo = base_algo

        X_all, feature_cols, num_cols, cat_cols = self._prepare_features(
            df,
            feature_spec=cfg.get("features", {}),
            task_name="hazard",
        )
        models = {}
        console_kv(
            "Hazard Room",
            {
                "rows": fmt_num(len(X_all)),
                "features": fmt_num(len(feature_cols)),
                "horizon_bins": fmt_num(horizon),
                "trials_per_bin": fmt_num(hpo_trials),
                "base_algo": base_algo,
            },
            style="yellow",
        )
        min_event_fraction = float(cfg.get("min_event_fraction", 0.005))
        eligible_total = 0
        for b in range(1, horizon + 1):
            y_probe = ((event == 1) & (tte <= b)).astype(int)
            if y_probe.mean() >= min_event_fraction:
                eligible_total += 1
        console_stage(
            "Hazard schedule",
            f"eligible_bins={eligible_total}/{horizon} min_event_fraction={min_event_fraction:.4f}",
            status="info",
        )
        hazard_started = time.perf_counter()
        processed_bins = 0

        # Optional challenger selection on a representative probe bin,
        # then reuse the winner for all hazard bins to control runtime.
        if bool(cfg.get("challenger_enabled", False)):
            probe_bin = int(cfg.get("challenger_probe_bin", min(12, horizon)))
            y_probe = ((event == 1) & (tte <= probe_bin)).astype(int)
            if np.unique(y_probe).size >= 2:
                probe_cfg = deepcopy(cfg)
                probe_cfg["algorithm"] = base_algo
                probe_cfg["hpo_trials"] = int(cfg.get("challenger_hpo_trials", max(2, hpo_trials)))
                _, probe_metrics = self._train_with_challengers(
                    X_df=X_all,
                    y=y_probe,
                    cfg=probe_cfg,
                    name=f"hazard_probe_b{probe_bin}",
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    default_algo=base_algo,
                )
                selected_algo = self._resolve_classifier_algo(
                    str(probe_metrics.get("selected_algorithm", base_algo)).lower()
                )
                console_stage(
                    "Hazard champion selected",
                    f"probe_bin={probe_bin} algo={selected_algo}",
                    status="ok",
                )
            else:
                console_stage(
                    "Hazard challenger skipped",
                    f"probe_bin={probe_bin} had single-class target",
                    status="warn",
                )

        default_params = self._default_params_for_algo(selected_algo, hpo_space)

        for b in range(1, horizon + 1):
            y_bin = ((event == 1) & (tte <= b)).astype(int)
            if y_bin.mean() < min_event_fraction:
                continue
            processed_bins += 1
            bin_started = time.perf_counter()
            if b == 1 or b == horizon or b % max(1, horizon // 6) == 0:
                console_stage(
                    f"Hazard bin {processed_bins}/{max(eligible_total, 1)}",
                    f"bar={b}/{horizon} event_rate={y_bin.mean():.4f} algo={selected_algo}",
                    status="info",
                )
            cw_bin, scale_pos_bin = self._make_class_weight(y_bin, cfg)

            def param_space(trial):
                if selected_algo in ("lightgbm", "xgboost"):
                    params = {
                        "n_estimators": trial.suggest_int(
                            "n_estimators", int(hpo_space.get("n_estimators", [150, 400])[0]),
                            int(hpo_space.get("n_estimators", [150, 400])[1]),
                        ),
                        "max_depth": trial.suggest_int(
                            "max_depth", int(hpo_space.get("max_depth", [3, 10])[0]),
                            int(hpo_space.get("max_depth", [3, 10])[1]),
                        ),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", float(hpo_space.get("learning_rate", [0.005, 0.12])[0]),
                            float(hpo_space.get("learning_rate", [0.005, 0.12])[1]),
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
                            "reg_alpha", float(hpo_space.get("reg_alpha", [0.0, 5.0])[0]),
                            float(hpo_space.get("reg_alpha", [0.0, 5.0])[1]),
                        ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda", float(hpo_space.get("reg_lambda", [0.0, 5.0])[0]),
                            float(hpo_space.get("reg_lambda", [0.0, 5.0])[1]),
                        ),
                    }
                    if selected_algo == "lightgbm":
                        params["num_leaves"] = trial.suggest_int(
                            "num_leaves", int(hpo_space.get("num_leaves", [15, 127])[0]),
                            int(hpo_space.get("num_leaves", [15, 127])[1]),
                        )
                    return params
                return {
                    "C": trial.suggest_float(
                        "C", float(hpo_space.get("C", [0.01, 10.0])[0]),
                        float(hpo_space.get("C", [0.01, 10.0])[1]),
                        log=True,
                    )
                }

            def build_estimator(params: Dict[str, Any]):
                if selected_algo == "lightgbm":
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
                        class_weight=cw_bin,
                    )
                if selected_algo == "xgboost":
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
                        scale_pos_weight=scale_pos_bin,
                    )
                return LogisticRegression(
                    C=float(params["C"]),
                    max_iter=400,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=cw_bin,
                )

            def objective(trial):
                params = param_space(trial)
                base = build_estimator(params)
                pre = self._build_preprocessor(num_cols, cat_cols, algo=selected_algo)
                model = Pipeline([("pre", pre), ("clf", base)])
                tscv = self._make_tscv(len(X_all), cv_splits)
                if tscv is None:
                    with self._suppress_low_signal_warnings():
                        model.fit(X_all, y_bin)
                        prob = self._positive_class_proba(model, X_all)
                        return -float(average_precision_score(y_bin, prob))
                scores = []
                for tr_idx, va_idx in tscv.split(X_all):
                    Xt, Xv = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                    yt, yv = y_bin[tr_idx], y_bin[va_idx]
                    with self._suppress_low_signal_warnings():
                        model.fit(Xt, yt)
                        prob = self._positive_class_proba(model, Xv)
                        pr = average_precision_score(yv, prob)
                    scores.append(pr)
                return -float(np.mean(scores))

            if cfg.get("hpo", {}).get("enabled", True) and optuna is not None and hpo_trials > 0:
                study = self._make_study(direction="minimize", cfg=cfg)
                if study is not None:
                    def _cb(study, trial):
                        if trial.value is None:
                            return
                        should_log = (
                            trial.number == 0
                            or (trial.number + 1) == hpo_trials
                            or ((trial.number + 1) % max(1, min(5, hpo_trials // 4 or 1)) == 0)
                        )
                        if should_log:
                            self._log_hpo_progress(
                                name=f"hazard_b{b}",
                                trial_number=trial.number + 1,
                                total_trials=hpo_trials,
                                trial_value=float(trial.value),
                                best_value=float(study.best_value),
                                started_at=bin_started,
                                direction="minimize",
                                extra={"objective": "neg_ap"},
                            )
                    with self._suppress_low_signal_warnings():
                        study.optimize(objective, n_trials=hpo_trials, show_progress_bar=False, callbacks=[_cb])
                    best_params = study.best_params
                else:
                    best_params = default_params
            else:
                best_params = default_params

            clf = build_estimator(best_params)
            pre = self._build_preprocessor(num_cols, cat_cols, algo=selected_algo)
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            with self._suppress_low_signal_warnings():
                pipe.fit(X_all, y_bin)
            models[b] = pipe
            total_elapsed = time.perf_counter() - hazard_started
            rate = processed_bins / max(total_elapsed, 1e-6)
            eta = (max(eligible_total, 1) - processed_bins) / max(rate, 1e-6)
            console_stage(
                "Hazard bin done",
                (
                    f"bar={b} progress={processed_bins}/{max(eligible_total, 1)} "
                    f"elapsed={fmt_seconds(total_elapsed)} eta={fmt_seconds(eta)} "
                    f"bin_elapsed={fmt_seconds(time.perf_counter() - bin_started)}"
                ),
                status="ok",
            )

        LOG.info(f"[ModelTrainer] Hazard model trained with {len(models)} bins.")
        return models, {
            "horizon_bars": horizon,
            "feature_cols": feature_cols,
            "selected_algorithm": selected_algo,
        }

    # ------------------------------------------------------------------
    # Quantile forecaster (LightGBM regressor)
    # ------------------------------------------------------------------
    def _train_quantile(self, df: pd.DataFrame, cfg: Dict[str, Any], prices: np.ndarray):
        quantiles = cfg.get("quantiles", [0.05, 0.1, 0.5, 0.9, 0.95])
        n_trials = int(cfg.get("hpo_trials", 20))
        cv_splits = int(cfg.get("cv_splits", 3))
        hpo_space = cfg.get("hpo_space", {})
        algo = self._resolve_quantile_algo()
        default_params = self._default_params_for_algo(algo, hpo_space)

        returns = pd.Series(prices, copy=False).pct_change().shift(-1)
        X_all, feature_cols, num_cols, cat_cols = self._prepare_features(
            df,
            feature_spec=cfg.get("features", {}),
            task_name="quantile",
        )
        valid_mask = returns.notna().values
        X_all = X_all.loc[valid_mask].reset_index(drop=True)
        returns = returns.loc[valid_mask].astype(float).values

        models = {}
        console_kv(
            "Quantile Room",
            {
                "rows": fmt_num(len(X_all)),
                "features": fmt_num(len(feature_cols)),
                "quantiles": ", ".join(str(q) for q in quantiles),
                "algo": algo,
                "trials": fmt_num(n_trials),
            },
            style="cyan",
        )
        quant_started = time.perf_counter()
        total_q = len(quantiles)

        for q_idx, q in enumerate(quantiles, start=1):
            LOG.info(f"[ModelTrainer] Training quantile model q={q}")
            q_name = f"quantile_{q}"
            q_t0 = time.perf_counter()
            console_stage(
                f"Quantile q={q} ({q_idx}/{total_q})",
                f"algo={algo}",
                status="info",
            )

            def param_space(trial):
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", int(hpo_space.get("n_estimators", [200, 600])[0]),
                        int(hpo_space.get("n_estimators", [200, 600])[1])
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
                if algo == "lightgbm":
                    params["num_leaves"] = trial.suggest_int(
                        "num_leaves", int(hpo_space.get("num_leaves", [31, 255])[0]),
                        int(hpo_space.get("num_leaves", [31, 255])[1])
                    )
                return params

            def build_quantile_model(params: Dict[str, Any]):
                if algo == "lightgbm":
                    return lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=int(params["n_estimators"]),
                        num_leaves=int(params["num_leaves"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                    )
                if algo == "xgboost":
                    return xgb.XGBRegressor(
                        objective="reg:quantileerror",
                        quantile_alpha=float(q),
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        tree_method="hist",
                    )
                return GradientBoostingRegressor(
                    loss="quantile",
                    alpha=float(q),
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    subsample=float(params["subsample"]),
                )

            def objective(trial):
                params = param_space(trial)
                base_model = build_quantile_model(params)
                pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
                model = Pipeline([("pre", pre), ("reg", base_model)])
                tscv = self._make_tscv(len(X_all), cv_splits)
                if tscv is None:
                    with self._suppress_low_signal_warnings():
                        model.fit(X_all, returns)
                        pred = model.predict(X_all)
                        e = returns - pred
                        return float(np.mean(np.maximum(q * e, (q - 1) * e)))
                losses = []
                for tr_idx, va_idx in tscv.split(X_all):
                    Xt, Xv = X_all.iloc[tr_idx], X_all.iloc[va_idx]
                    yt, yv = returns[tr_idx], returns[va_idx]
                    with self._suppress_low_signal_warnings():
                        model.fit(Xt, yt)
                        pred = model.predict(Xv)
                        e = yv - pred
                        pinball = np.mean(np.maximum(q * e, (q - 1) * e))
                    losses.append(pinball)
                return float(np.mean(losses))

            if optuna is None or n_trials <= 0:
                LOG.warning("[ModelTrainer] Optuna unavailable for quantile q=%s; using default params=%s", q, default_params)
                best_params = default_params
                console_stage(
                    f"Quantile q={q} default params",
                    f"trials={n_trials} -> using midpoint defaults",
                    status="warn",
                )
            else:
                study = self._make_study(direction="minimize", cfg=cfg)
                if study is None:
                    best_params = default_params
                    base_final = build_quantile_model(best_params)
                    pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
                    pipe = Pipeline([("pre", pre), ("reg", base_final)])
                    with self._suppress_low_signal_warnings():
                        pipe.fit(X_all, returns)
                    models[f"q_{q}"] = pipe
                    console_stage(
                        f"Quantile q={q} ready",
                        f"elapsed={fmt_seconds(time.perf_counter() - q_t0)}",
                        status="ok",
                    )
                    continue
                def _cb(study, trial):
                    if trial.value is None:
                        return
                    should_log = (
                        trial.number == 0
                        or (trial.number + 1) == n_trials
                        or ((trial.number + 1) % max(1, min(5, n_trials // 4 or 1)) == 0)
                    )
                    if should_log:
                        self._log_hpo_progress(
                            name=q_name,
                            trial_number=trial.number + 1,
                            total_trials=n_trials,
                            trial_value=float(trial.value),
                            best_value=float(study.best_value),
                            started_at=q_t0,
                            direction="minimize",
                            extra={"objective": "pinball"},
                        )

                with self._suppress_low_signal_warnings():
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_cb])
                best_params = study.best_params
                self._log_hpo_summary(
                    q_name,
                    float(study.best_value),
                    best_params,
                    q_t0,
                    metric_label="best_loss",
                    extra={"objective": "pinball"},
                )

            base_final = build_quantile_model(best_params)
            pre = self._build_preprocessor(num_cols, cat_cols, algo=algo)
            pipe = Pipeline([("pre", pre), ("reg", base_final)])
            with self._suppress_low_signal_warnings():
                pipe.fit(X_all, returns)
            models[f"q_{q}"] = pipe
            console_stage(
                f"Quantile q={q} ready",
                f"elapsed={fmt_seconds(time.perf_counter() - q_t0)}",
                status="ok",
            )
            total_elapsed = time.perf_counter() - quant_started
            rate = q_idx / max(total_elapsed, 1e-6)
            eta = (total_q - q_idx) / max(rate, 1e-6)
            console_stage(
                "Quantile progress",
                (
                    f"{q_idx}/{total_q} done "
                    f"elapsed={fmt_seconds(total_elapsed)} eta={fmt_seconds(eta)}"
                ),
                status="info",
            )

        return models, {"quantiles": quantiles, "features": feature_cols}
