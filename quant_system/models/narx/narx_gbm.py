"""
NARX-style forecaster using GBM (LightGBM quantile if available, HistGBR fallback).

 - Target: continuous (e.g., forward log-return). Default: ret_fwd_{horizon_bars}.
 - Lags: autoregressive lags on realized returns + lagged exogenous features.
 - CV: time-series splits with optional purge.
 - Saves meta, metrics, and model artifacts via joblib.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional LightGBM (preferred for quantile heads)
try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False


# ------------------------ helpers ------------------------ #
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.maximum(q * e, (q - 1) * e).mean())


def logret(s: pd.Series) -> pd.Series:
    return np.log(s).diff().fillna(0.0)


def make_forward_logret(close: pd.Series, H: int) -> pd.Series:
    return np.log(close.shift(-H) / close)


def build_time_blocks(n: int, n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


# ------------------------ config ------------------------ #
@dataclass
class NARXGBMConfig:
    horizon_bars: int = 8                 # predict next H bars
    y_col: Optional[str] = None           # if None -> ret_fwd_{H}
    y_lags: List[int] = (1, 2, 4, 8, 16)  # AR lags on realized return
    x_lags: List[int] = (1, 2, 4, 8)      # lags for exogenous features
    exog_cols: Optional[List[str]] = None # exogenous feature list
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    n_splits: int = 5
    purge_bars: int = 96                  # purge between train/val
    seed: int = 42
    lgbm_params: Dict = None              # optional override
    hgb_params: Dict = None               # fallback params


# ------------------------ dataset builder ------------------------ #
class NARXDesign:
    def __init__(self, cfg: NARXGBMConfig):
        self.cfg = cfg

    def _lag_frame(self, df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in cols:
            if c not in df.columns:
                continue
            for L in lags:
                out[f"{c}_lag{L}"] = df[c].shift(L)
        return out

    def _ensure_target(self, df: pd.DataFrame) -> pd.Series:
        if self.cfg.y_col and self.cfg.y_col in df.columns:
            return df[self.cfg.y_col].astype(float)
        y = make_forward_logret(df["close"].astype(float), self.cfg.horizon_bars)
        return y.rename(f"ret_fwd_{self.cfg.horizon_bars}")

    def build(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        df = df.copy().sort_values("dt").reset_index(drop=True)

        # target
        y = self._ensure_target(df)

        # autoregressive lags (use realized short return)
        ar_src = logret(df["close"].astype(float))
        ar_df = pd.DataFrame({"ar": ar_src})
        X_y = self._lag_frame(ar_df, ["ar"], list(self.cfg.y_lags))

        # exogenous lags
        exog_cols = self.cfg.exog_cols
        if exog_cols is None:
            exog_cols = [
                c
                for c in [
                    "atr",
                    "volume",
                    "dollar_volume",
                    "ema_fast_15m",
                    "ema_slow_15m",
                    "ema_rel_1h",
                    "absorption_score",
                    "session_weight",
                    "bias_6h",
                    "bias_12h",
                    "bos_flag_1h",
                    "fvg_ctx_weight",
                    "fvg_ctx_dir",
                    "regime_state_id",
                ]
                if c in df.columns
            ]
            self.cfg.exog_cols = exog_cols
        X_exo = self._lag_frame(df, exog_cols, list(self.cfg.x_lags))

        X = pd.concat([X_y, X_exo], axis=1)
        data = pd.concat([X, y.rename("y")], axis=1).dropna()
        feat_list = [c for c in data.columns if c != "y"]
        return data[feat_list], data["y"].astype(float), feat_list


# ------------------------ trainer ------------------------ #
class NARXGBM:
    """
    Quantile LightGBM if available; else HistGradientBoostingRegressor (point forecast).
    Saves meta.json, models (*.joblib), and stores CV metrics in self.metrics_.
    """

    def __init__(self, cfg: NARXGBMConfig):
        self.cfg = cfg
        self.design = NARXDesign(cfg)
        self.models_: Dict[str, object] = {}
        self.features_: List[str] = []
        self.metrics_: Dict = {}
        self.kind_: str = "point"

    def _purge_train(self, idx: np.ndarray) -> np.ndarray:
        if self.cfg.purge_bars <= 0:
            return idx
        cut = idx.max() - self.cfg.purge_bars
        return idx[idx <= cut]

    def _fit_lgbm_quantile(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
        models = {}
        params_base = dict(
            objective="quantile",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_samples=80,
            random_state=self.cfg.seed,
            n_jobs=-1,
        )
        if self.cfg.lgbm_params:
            params_base.update(self.cfg.lgbm_params)

        tscv = build_time_blocks(len(X), self.cfg.n_splits)
        splits = list(tscv.split(X))
        tr_idx, va_idx = splits[-1]
        tr_idx = self._purge_train(tr_idx)
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

        fold_metrics = []
        for q in self.cfg.quantiles:
            m = lgb.LGBMRegressor(**params_base, alpha=q)
            m.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                eval_metric=lambda yt, yp: ("pinball", pinball_loss(yt, yp, q), False),
                callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
            )
            p = m.predict(Xva)
            qloss = pinball_loss(yva.values, p, q)
            fold_metrics.append({f"q{int(q*100)}_pinball": float(qloss)})
            models[f"q{int(q*100)}"] = m

        if "q50" in models:
            mae = mean_absolute_error(yva.values, models["q50"].predict(Xva))
            fold_metrics.append({"q50_mae": float(mae)})

        self.metrics_["val_last_fold"] = fold_metrics
        return models

    def _fit_hgb_point(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
        params = dict(
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=80,
            l2_regularization=0.0,
            random_state=self.cfg.seed,
        )
        if self.cfg.hgb_params:
            params.update(self.cfg.hgb_params)

        tscv = build_time_blocks(len(X), self.cfg.n_splits)
        fold_mae = []
        for tr_idx, va_idx in tscv.split(X):
            tr_idx = self._purge_train(tr_idx)
            Xt, yt = X.iloc[tr_idx], y.iloc[tr_idx]
            Xv, yv = X.iloc[va_idx], y.iloc[va_idx]

            num_cols = Xt.columns.tolist()
            pre = ColumnTransformer([("num", StandardScaler(with_mean=False), num_cols)], remainder="drop")
            pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(**params))])
            pipe.fit(Xt, yt)
            pred = pipe.predict(Xv)
            fold_mae.append(mean_absolute_error(yv, pred))

        self.metrics_["cv_mae"] = float(np.mean(fold_mae))

        num_cols = X.columns.tolist()
        pre = ColumnTransformer([("num", StandardScaler(with_mean=False), num_cols)], remainder="drop")
        pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingRegressor(**params))])
        pipe.fit(X, y)
        return {"point": pipe}

    def fit(self, df_features: pd.DataFrame) -> Dict:
        X, y, feats = self.design.build(df_features)
        self.features_ = feats

        if _HAS_LGBM:
            self.models_ = self._fit_lgbm_quantile(X, y)
            self.kind_ = "quantile"
        else:
            self.models_ = self._fit_hgb_point(X, y)
            self.kind_ = "point"

        return dict(kind=self.kind_, features=len(self.features_), metrics=self.metrics_)

    # ---------------- inference ---------------- #
    def predict_point(self, Xrow: pd.DataFrame) -> float:
        if self.kind_ == "quantile":
            m = self.models_["q50"]
            return float(m.predict(Xrow)[0])
        return float(self.models_["point"].predict(Xrow)[0])

    def predict_interval(self, Xrow: pd.DataFrame, q_low=0.1, q_high=0.9) -> Tuple[float, float]:
        if self.kind_ != "quantile":
            p = self.predict_point(Xrow)
            return p, p
        ml = self.models_.get(f"q{int(q_low*100)}", self.models_["q50"])
        mh = self.models_.get(f"q{int(q_high*100)}", self.models_["q50"])
        return float(ml.predict(Xrow)[0]), float(mh.predict(Xrow)[0])

    # ---------------- persistence ---------------- #
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "cfg": asdict(self.cfg),
            "features": self.features_,
            "kind": self.kind_,
            "metrics": self.metrics_,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        for name, model in self.models_.items():
            joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))

    @staticmethod
    def load(model_dir: str) -> "NARXGBM":
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = NARXGBMConfig(**meta["cfg"])
        obj = NARXGBM(cfg)
        obj.kind_ = meta["kind"]
        obj.features_ = meta["features"]
        obj.metrics_ = meta.get("metrics", {})
        models = {}
        for f in os.listdir(model_dir):
            if f.endswith(".joblib"):
                models[f.replace(".joblib", "")] = joblib.load(os.path.join(model_dir, f))
        obj.models_ = models
        return obj

    def build_live_row(self, df_tail: pd.DataFrame) -> pd.DataFrame:
        X, _, feats = self.design.build(df_tail.copy())
        X = X.iloc[[-1]]
        for c in self.features_:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.features_]
        return X
