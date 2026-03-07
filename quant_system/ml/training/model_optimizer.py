"""
Bayesian hyperparameter optimizer using Optuna for
time-series ML models (LightGBM/XGBoost/Logistic).
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

from quant_system.utils.logger import log
from quant_system.utils.pandas_compat import ensure_stringmethods_alias


class ModelOptimizer:
    """
    Bayesian HPO for specialist models with time-series evaluation.
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        self.cfg = model_cfg
        self.n_trials = model_cfg.get("hpo_trials", 40)
        self.n_splits = model_cfg.get("cv_splits", 4)
        log("ModelOptimizer ready.")

    def _build_model(self, params: Dict[str, Any]):
        """Instantiate model based on params."""
        algo = self.cfg["algorithm"].lower()

        if algo == "logistic":
            return LogisticRegression(
                C=params["C"],
                max_iter=self.cfg.get("max_iter", 500),
                penalty="l2",
                solver="lbfgs",
            )

        elif algo == "lightgbm":
            ensure_stringmethods_alias()
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=int(params["n_estimators"]),
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
            )

        elif algo == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                eta=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                eval_metric="logloss",
                tree_method="hist",
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

    def _param_space(self, trial: Any) -> Dict[str, Any]:
        """Define search space per algorithm."""
        algo = self.cfg["algorithm"].lower()

        if algo == "logistic":
            return {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            }

        elif algo == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }

        elif algo == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }

        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

    def _timeseries_score(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Walk-forward evaluation. Optimization metric is:
        - PR-AUC for imbalanced binary tasks (liq-flow, bos-cont, momo, EOP, EDP)
        - AUC fallback if PR fails (rare)
        """

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            Xt, Xv = X[train_idx], X[val_idx]
            yt, yv = y[train_idx], y[val_idx]

            model.fit(Xt, yt)
            prob = model.predict_proba(Xv)[:, 1]

            try:
                pr = average_precision_score(yv, prob)
            except:
                pr = 0.0

            try:
                auc = roc_auc_score(yv, prob)
            except:
                auc = 0.5

            score = pr if pr > 0 else auc
            scores.append(score)

        return float(np.mean(scores))

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run Bayesian HPO, return best param set.
        """

        log("Starting Bayesian HPO...")
        if optuna is None:
            log("Optuna unavailable. Falling back to empty/default parameter set.")
            return {}

        def objective(trial: Any):
            params = self._param_space(trial)
            model = self._build_model(params)
            score = self._timeseries_score(model, X, y)
            return -score  # minimize negative score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        best_params = study.best_params
        log(f"HPO complete. Best params: {best_params}")

        return best_params
