"""
TCN specialist trainer for binary label tasks with Optuna HPO and time-series CV.

This module is intentionally independent from the tree-model trainer so we can
benchmark deep learning vs current LightGBM/XGBoost pipelines on the same
engineered features and labels.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from quant_system.ml.predict.empirical_calibrator import EmpiricalCalibrator
from quant_system.utils.logger import console_stage, fmt_num, fmt_seconds, get_logger

LOG = get_logger("tcn_trainer")


TARGET_COLUMN_MAP: Dict[str, str] = {
    "liq_flow": "label_liq_flow",
    "bos_cont": "label_bos_cont",
    "momo": "label_momo",
    "flow_1h": "label_flow_1h",
    "eop": "label_eop",
    "edp": "label_edp",
}


DEFAULT_TCN_CFG: Dict[str, Any] = {
    "random_seed": 42,
    "hpo_trials": 20,
    "hpo_resume": True,
    "hpo_storage": None,
    "hpo_study_name": None,
    "hpo_pruner_warmup_steps": 1,
    "hpo_gc_after_trial": True,
    "hpo_log_every_n_trials": 1,
    "hpo_adaptive_stop": False,
    "hpo_adaptive_min_completed_trials": 12,
    "hpo_adaptive_no_improve_trials": 8,
    "hpo_adaptive_min_delta": 0.001,
    "fold_heartbeat": True,
    "epoch_heartbeat_seconds": 120,
    "progress_snapshot_filename": "hpo_progress.json",
    "progress_events_filename": "hpo_progress.ndjson",
    "cv_splits": 4,
    "embargo_bars": 2,
    "feature_source": "tree_manifest",  # tree_manifest | auto
    "calibrator": "auto",  # auto | platt | isotonic | empirical | none
    "holdout_frac": 0.15,
    "acceptance": {
        "enabled": True,
        "holdout_frac": 0.10,
        "min_rows": 4096,
        "max_score_drop": 0.05,
        "min_score": None,
        "min_precision_at_threshold": 0.10,
    },
    "stability": {
        "enabled": True,
        "seeds": [42, 52, 62],
        "max_std": 0.02,
        "max_drop_vs_best": 0.04,
    },
    "threshold_tuning": {
        "enabled": True,
        "metric": "f1",   # f1 | precision | recall
        "beta": 1.0,       # reserved for future f-beta support
        "min_precision": 0.10,
        "min_recall": 0.01,
        "default_threshold": 0.50,
        "max_candidates": 256,
    },
    "feature_dominance_audit": {
        "enabled": True,
        "min_rows": 4096,
        "sample_rows": 20000,
        "random_state": 42,
        "min_top_drop": 0.01,
        "dominance_share_warn": 0.55,
        "monitor_features_by_target": {
            "flow_1h": ["flow_age_bars_1h", "flow_ok_1h", "flow_signal_1h"],
        },
    },
    "device": "cpu",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "mkldnn_enabled": False,
    "preprocessing": {
        "num_imputer": "median",
        "cat_imputer": "most_frequent",
        "scaler": "standard",  # standard | robust | none
        "outlier_clip": True,
        "clip_quantiles": [0.005, 0.995],
    },
    "hpo_space": {
        "lookback": [24, 96],
        "levels": [1, 3],
        "channels": [16, 64],
        "kernel_size": [2, 5],
        "dropout": [0.0, 0.35],
        "lr": [1e-4, 3e-3],
        "weight_decay": [1e-8, 1e-3],
        "batch_size_choices": [64, 128, 256],
        "max_epochs": [8, 24],
        "patience": [3, 7],
    },
}


def merge_tcn_cfg(
    *,
    models_yaml: Dict[str, Any],
    target: str,
    trials_override: Optional[int] = None,
    cv_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Merge defaults + models.yaml tcn_training + task overrides + CLI overrides.
    """
    cfg = deepcopy(DEFAULT_TCN_CFG)
    tcn_root = deepcopy((models_yaml or {}).get("tcn_training", {}))
    if isinstance(tcn_root.get("default"), dict):
        cfg = _deep_update(cfg, tcn_root.get("default", {}))
    if isinstance(tcn_root.get("overrides"), dict) and isinstance(tcn_root["overrides"].get(target), dict):
        cfg = _deep_update(cfg, tcn_root["overrides"][target])
    if trials_override is not None:
        cfg["hpo_trials"] = int(trials_override)
    if cv_override is not None:
        cfg["cv_splits"] = int(cv_override)
    return cfg


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuantileClipper:
    """Leak-safe winsorization fit on train folds only."""

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


class SequenceIndexDataset(Dataset):
    """
    Creates rolling windows ending at provided indices.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        end_indices: np.ndarray,
        lookback: int,
    ):
        self.X = X
        self.y = y
        self.end_indices = end_indices.astype(int)
        self.lookback = int(lookback)

    def __len__(self) -> int:
        return int(len(self.end_indices))

    def __getitem__(self, idx: int):
        end_idx = int(self.end_indices[idx])
        start_idx = end_idx - self.lookback + 1
        seq = torch.tensor(self.X[start_idx : end_idx + 1], dtype=torch.float32)
        if self.y is None:
            return seq
        target = torch.tensor(float(self.y[end_idx]), dtype=torch.float32)
        return seq, target


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + residual)


class TCNBinaryClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = int(input_dim)
        for i, out_ch in enumerate(channels):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=int(out_ch),
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )
            in_ch = int(out_ch)
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, F] -> [B, F, T]
        x = x_seq.transpose(1, 2)
        z = self.tcn(x)  # [B, C, T]
        last = z[:, :, -1]  # [B, C]
        logits = self.head(last).squeeze(-1)  # [B]
        return logits


class TCNCalibratedModel:
    """
    Joblib-persistable prediction bundle for TCN specialist model.
    """

    def __init__(
        self,
        *,
        feature_cols: List[str],
        lookback: int,
        preprocessor: ColumnTransformer,
        model_params: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
        calibrator: Optional[Any] = None,
        infer_batch_size: int = 512,
        decision_threshold: float = 0.5,
    ):
        self.feature_cols = list(feature_cols)
        self.lookback = int(lookback)
        self.preprocessor = preprocessor
        self.model_params = deepcopy(model_params)
        self.state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        self.calibrator = calibrator
        self.infer_batch_size = int(infer_batch_size)
        self.decision_threshold = float(decision_threshold)
        self._model: Optional[TCNBinaryClassifier] = None
        self._build_model()

    def _build_model(self) -> None:
        model = TCNBinaryClassifier(**self.model_params)
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        self._model = model

    def __getstate__(self):
        payload = self.__dict__.copy()
        payload["_model"] = None
        return payload

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_model()

    @staticmethod
    def _ensure_dense_float32(X_mat: Any) -> np.ndarray:
        if hasattr(X_mat, "toarray"):
            X_mat = X_mat.toarray()
        return np.asarray(X_mat, dtype=np.float32)

    def _apply_calibrator(self, p_raw: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p_raw
        if hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        if hasattr(self.calibrator, "predict"):
            return self.calibrator.predict(p_raw)
        return np.asarray([float(self.calibrator(v)) for v in p_raw], dtype=float)

    def _predict_raw_proba(self, X_mat: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._build_model()
        assert self._model is not None
        n_rows = int(X_mat.shape[0])
        proba = np.full(n_rows, 0.5, dtype=float)
        if n_rows < self.lookback:
            return proba

        end_indices = np.arange(self.lookback - 1, n_rows, dtype=int)
        dataset = SequenceIndexDataset(X_mat, None, end_indices, self.lookback)
        loader = DataLoader(dataset, batch_size=self.infer_batch_size, shuffle=False)
        probs_valid: List[np.ndarray] = []
        with torch.no_grad():
            for xb in loader:
                logits = self._model(xb)
                p = torch.sigmoid(logits).cpu().numpy()
                probs_valid.append(p)
        merged = np.concatenate(probs_valid) if probs_valid else np.empty(0, dtype=float)
        proba[end_indices] = merged
        return proba

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in X_df.columns]
        if missing:
            raise KeyError(f"Missing required features for TCN model: {missing[:8]}")
        X_used = X_df[self.feature_cols].copy()
        X_mat = self.preprocessor.transform(X_used)
        X_mat = self._ensure_dense_float32(X_mat)
        p_raw = self._predict_raw_proba(X_mat)
        p = np.clip(self._apply_calibrator(p_raw), 1e-6, 1.0 - 1e-6)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X_df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        p = self.predict_proba(X_df)[:, 1]
        thr = self.decision_threshold if threshold is None else float(threshold)
        return (p >= thr).astype(int)


class TCNSpecialistTrainer:
    def __init__(
        self,
        *,
        asset: str,
        target: str,
        config: Dict[str, Any],
        tree_manifest_path: Optional[Path] = None,
    ):
        if target not in TARGET_COLUMN_MAP:
            raise ValueError(f"Unsupported TCN target: {target}")
        self.asset = str(asset)
        self.target = str(target)
        self.label_col = TARGET_COLUMN_MAP[target]
        self.cfg = deepcopy(config)
        self.tree_manifest_path = tree_manifest_path
        self.progress_dir = Path(str(self.cfg.get("artifact_root", "")).strip() or ".")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.progress_snapshot_path = self.progress_dir / str(self.cfg.get("progress_snapshot_filename", "hpo_progress.json"))
        self.progress_events_path = self.progress_dir / str(self.cfg.get("progress_events_filename", "hpo_progress.ndjson"))
        self._configure_external_logging()
        self._set_runtime_knobs()

    @staticmethod
    def _configure_external_logging() -> None:
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception:
            pass

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_snapshot_started_at(self) -> Optional[float]:
        if not self.progress_snapshot_path.exists():
            return None
        try:
            payload = json.loads(self.progress_snapshot_path.read_text(encoding="utf-8"))
            val = payload.get("run_started_at_epoch")
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def _write_progress_snapshot(self, payload: Dict[str, Any], *, append_event: bool = True) -> None:
        body = dict(payload)
        body["updated_at_utc"] = self._utc_now_iso()
        try:
            self.progress_snapshot_path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            LOG.warning("[TCN] failed to write progress snapshot %s: %s", self.progress_snapshot_path, exc)
        if append_event:
            try:
                with self.progress_events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(body, sort_keys=True))
                    f.write("\n")
            except Exception as exc:
                LOG.warning("[TCN] failed to append progress event %s: %s", self.progress_events_path, exc)

    def _set_runtime_knobs(self) -> None:
        seed = int(self.cfg.get("random_seed", 42))
        _set_global_seed(seed)
        # Native thread pools on macOS (BLAS/OpenMP + torch) can segfault under
        # high parallelism; keep defaults conservative unless explicitly overridden.
        n_threads = int(self.cfg.get("torch_num_threads", 1))
        if n_threads <= 0:
            n_threads = 1
        torch.set_num_threads(n_threads)

        interop_threads = int(self.cfg.get("torch_num_interop_threads", 1))
        if interop_threads > 0:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                # Can only be set once per process in some torch builds.
                pass

        if str(self.cfg.get("device", "cpu")).lower() == "cpu":
            try:
                torch.backends.mkldnn.enabled = bool(self.cfg.get("mkldnn_enabled", False))
            except Exception:
                pass

        LOG.info(
            "[TCN] runtime knobs | torch_threads=%s interop=%s mkldnn=%s omp=%s mkl=%s openblas=%s vecLib=%s",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            bool(getattr(torch.backends.mkldnn, "enabled", False)),
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get("MKL_NUM_THREADS"),
            os.environ.get("OPENBLAS_NUM_THREADS"),
            os.environ.get("VECLIB_MAXIMUM_THREADS"),
        )

    @staticmethod
    def _is_prediction_column(col: str) -> bool:
        if col.startswith("label_"):
            return True
        if col.startswith("prob_"):
            return True
        if col in {"hazard_event", "hazard_time", "hazard", "hazard_score", "prob_meta", "prob_confluence"}:
            return True
        if col.startswith("hazard_curve"):
            return True
        if col == "quantiles" or col.startswith("quantiles_"):
            return True
        if col.startswith("p_"):
            return True
        return False

    def _default_feature_cols(self, df: pd.DataFrame) -> List[str]:
        drop_cols = {"dt", "timestamp"}
        cols = [
            c
            for c in df.columns
            if c not in drop_cols and not self._is_prediction_column(c) and c != self.label_col
        ]
        return cols

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        source = str(self.cfg.get("feature_source", "tree_manifest")).lower()
        if source == "tree_manifest":
            path = self.tree_manifest_path
            if path is not None and path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    selected = (
                        payload.get("metrics", {})
                        .get("by_model", {})
                        .get(self.target, {})
                        .get("selected_feature_cols", [])
                    )
                    selected = [c for c in selected if c in df.columns]
                    if selected:
                        return selected
                except Exception as exc:
                    LOG.warning("[TCN] Failed to read tree manifest %s: %s", path, exc)
        return self._default_feature_cols(df)

    @staticmethod
    def _ensure_dense_float32(X_mat: Any) -> np.ndarray:
        if hasattr(X_mat, "toarray"):
            X_mat = X_mat.toarray()
        return np.asarray(X_mat, dtype=np.float32)

    def _build_preprocessor(self, X_df: pd.DataFrame) -> ColumnTransformer:
        prep_cfg = deepcopy(self.cfg.get("preprocessing", {}))
        num_imputer = str(prep_cfg.get("num_imputer", "median")).lower()
        cat_imputer = str(prep_cfg.get("cat_imputer", "most_frequent")).lower()
        scaler = str(prep_cfg.get("scaler", "standard")).lower()
        clip_enabled = bool(prep_cfg.get("outlier_clip", True))
        clip_q = prep_cfg.get("clip_quantiles", [0.005, 0.995]) or [0.005, 0.995]
        try:
            low_q = float(clip_q[0])
            high_q = float(clip_q[1])
        except Exception:
            low_q, high_q = 0.005, 0.995

        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]

        num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=num_imputer))]
        if clip_enabled:
            num_steps.append(("clip", QuantileClipper(low_q, high_q)))
        if scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        cat_steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=cat_imputer)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=num_steps), num_cols),
                ("cat", Pipeline(steps=cat_steps), cat_cols),
            ],
            remainder="drop",
        )

    @staticmethod
    def _binary_scores(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
        y = np.asarray(y_true, dtype=int)
        prob = np.asarray(p, dtype=float)
        if len(y) == 0:
            return 0.5, 0.5, 0.5
        if np.unique(y).size < 2:
            try:
                auc = float(roc_auc_score(y, prob))
            except Exception:
                auc = 0.5
            return 0.5, auc, auc
        try:
            ap = float(average_precision_score(y, prob))
        except Exception:
            ap = 0.5
        try:
            auc = float(roc_auc_score(y, prob))
        except Exception:
            auc = 0.5
        score = ap if np.isfinite(ap) else auc
        return ap, auc, score

    @staticmethod
    def _build_end_indices(raw_indices: np.ndarray, lookback: int, max_train_end: Optional[int] = None) -> np.ndarray:
        idx = np.asarray(raw_indices, dtype=int)
        idx = idx[idx >= (lookback - 1)]
        if max_train_end is not None:
            idx = idx[idx <= int(max_train_end)]
        return idx

    @staticmethod
    def _count_completed_trials(study: optuna.Study) -> int:
        done_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
        return int(sum(1 for tr in study.trials if tr.state in done_states))

    @staticmethod
    def _scan_best_state(study: optuna.Study) -> Tuple[Optional[float], int]:
        """
        Returns:
          - best completed trial value seen so far (or None)
          - completed-trial count when that best was last improved
        """
        done_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
        done = 0
        best_val: Optional[float] = None
        done_at_best = 0
        for trial in sorted(study.trials, key=lambda tr: tr.number):
            if trial.state not in done_states:
                continue
            done += 1
            if trial.state == TrialState.COMPLETE and trial.value is not None:
                trial_val = float(trial.value)
                if best_val is None or trial_val > best_val:
                    best_val = trial_val
                    done_at_best = done
        return best_val, int(done_at_best)

    def _split_dev_acceptance(self, X_df: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame], Optional[np.ndarray], int]:
        acc_cfg = deepcopy(self.cfg.get("acceptance", {}))
        if not bool(acc_cfg.get("enabled", True)):
            return X_df, y, None, None, 0
        n_rows = int(len(X_df))
        holdout_frac = float(acc_cfg.get("holdout_frac", 0.10))
        min_rows = max(int(acc_cfg.get("min_rows", 4096)), 512)
        holdout_rows = max(int(n_rows * holdout_frac), min_rows)
        holdout_rows = min(holdout_rows, max(n_rows // 3, 0))
        if holdout_rows < 512:
            return X_df, y, None, None, 0
        if (n_rows - holdout_rows) < 4096:
            return X_df, y, None, None, 0

        cut = n_rows - holdout_rows
        X_dev = X_df.iloc[:cut].copy()
        y_dev = y[:cut]
        X_acc = X_df.iloc[cut:].copy()
        y_acc = y[cut:]
        return X_dev, y_dev, X_acc, y_acc, int(holdout_rows)

    def _evaluate_binary_predictions(self, y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
        y = np.asarray(y_true, dtype=int)
        prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        ap, auc, score = self._binary_scores(y, prob)
        try:
            brier = float(brier_score_loss(y, prob))
        except Exception:
            brier = float("nan")
        y_hat = (prob >= float(threshold)).astype(int)
        precision = float(precision_score(y, y_hat, zero_division=0))
        recall = float(recall_score(y, y_hat, zero_division=0))
        f1 = float(f1_score(y, y_hat, zero_division=0))
        return {
            "score": float(score),
            "ap": float(ap),
            "auc": float(auc),
            "brier": brier,
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
            "positive_rate_at_threshold": float(np.mean(y_hat) if len(y_hat) else 0.0),
            "threshold": float(threshold),
        }

    def _feature_dominance_audit(
        self,
        model: Any,
        X_df: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        cfg = deepcopy(self.cfg.get("feature_dominance_audit", {}))
        if not bool(cfg.get("enabled", False)):
            return {}

        monitor_map = cfg.get("monitor_features_by_target", {})
        monitor_features: List[str] = []
        if isinstance(monitor_map, dict):
            raw = monitor_map.get(self.target, [])
            if isinstance(raw, list):
                monitor_features = [str(x) for x in raw if str(x) in X_df.columns]
        if not monitor_features:
            return {}

        min_rows = max(int(cfg.get("min_rows", 4096)), 256)
        if len(X_df) < min_rows:
            return {}
        if pd.Series(y).nunique() < 2:
            return {}

        sample_rows = max(int(cfg.get("sample_rows", 20000)), min_rows)
        random_state = int(cfg.get("random_state", 42))
        min_top_drop = float(cfg.get("min_top_drop", 0.01))
        warn_share = float(cfg.get("dominance_share_warn", 0.55))

        if len(X_df) > sample_rows:
            X_eval = X_df.tail(sample_rows).copy()
            y_eval = y[-sample_rows:]
        else:
            X_eval = X_df.copy()
            y_eval = y

        try:
            p_base = model.predict_proba(X_eval)[:, 1]
            base_ap = float(average_precision_score(y_eval, p_base))
        except Exception as exc:
            LOG.warning("[TCN] dominance audit skipped for %s: %s", self.target, exc)
            return {}

        rng = np.random.default_rng(random_state)
        rows: List[Dict[str, Any]] = []
        for idx, feat in enumerate(monitor_features):
            X_perm = X_eval.copy()
            values = X_perm[feat].to_numpy()
            X_perm[feat] = values[rng.permutation(len(values))]
            try:
                p_perm = model.predict_proba(X_perm)[:, 1]
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
                "[TCN] dominance audit %s top_feature=%s ap_drop=%.4f share=%.3f (threshold=%.3f)",
                self.target,
                top["feature"],
                float(top["ap_drop"]),
                float(dominance_share),
                float(warn_share),
            )

        return {
            "enabled": True,
            "target": self.target,
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

    def _tune_threshold(self, y_true: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
        cfg = deepcopy(self.cfg.get("threshold_tuning", {}))
        metric = str(cfg.get("metric", "f1")).lower()
        min_precision = float(cfg.get("min_precision", 0.10))
        min_recall = float(cfg.get("min_recall", 0.01))
        default_thr = float(cfg.get("default_threshold", 0.50))
        max_candidates = max(int(cfg.get("max_candidates", 256)), 16)

        y = np.asarray(y_true, dtype=int)
        prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        if len(y) == 0 or not bool(cfg.get("enabled", True)):
            metrics = self._evaluate_binary_predictions(y, prob, default_thr)
            return {"metric": metric, "threshold": default_thr, "best_value": metrics.get("f1_at_threshold", 0.0), "metrics": metrics}

        candidates = np.unique(prob)
        if candidates.size > max_candidates:
            qs = np.linspace(0.01, 0.99, max_candidates)
            candidates = np.unique(np.quantile(prob, qs))
        candidates = np.clip(candidates, 1e-4, 1.0 - 1e-4)
        candidates = np.unique(np.r_[candidates, default_thr])

        best_thr = default_thr
        best_value = -np.inf
        best_metrics = self._evaluate_binary_predictions(y, prob, default_thr)

        for thr in candidates:
            metrics = self._evaluate_binary_predictions(y, prob, float(thr))
            precision = float(metrics["precision_at_threshold"])
            recall = float(metrics["recall_at_threshold"])
            if precision < min_precision or recall < min_recall:
                continue
            if metric == "precision":
                value = precision
            elif metric == "recall":
                value = recall
            else:
                value = float(metrics["f1_at_threshold"])
            if value > best_value:
                best_value = value
                best_thr = float(thr)
                best_metrics = metrics

        if not np.isfinite(best_value):
            if metric == "precision":
                best_value = float(best_metrics["precision_at_threshold"])
            elif metric == "recall":
                best_value = float(best_metrics["recall_at_threshold"])
            else:
                best_value = float(best_metrics["f1_at_threshold"])

        return {
            "metric": metric,
            "threshold": float(best_thr),
            "best_value": float(best_value),
            "constraints": {
                "min_precision": min_precision,
                "min_recall": min_recall,
            },
            "metrics": best_metrics,
        }

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        hpo = deepcopy(self.cfg.get("hpo_space", {}))
        lookback = trial.suggest_int("lookback", int(hpo.get("lookback", [24, 96])[0]), int(hpo.get("lookback", [24, 96])[1]))
        levels = trial.suggest_int("levels", int(hpo.get("levels", [1, 3])[0]), int(hpo.get("levels", [1, 3])[1]))
        channels = trial.suggest_int(
            "channels",
            int(hpo.get("channels", [16, 64])[0]),
            int(hpo.get("channels", [16, 64])[1]),
            step=16,
        )
        kernel_size = trial.suggest_int(
            "kernel_size",
            int(hpo.get("kernel_size", [2, 5])[0]),
            int(hpo.get("kernel_size", [2, 5])[1]),
        )
        if kernel_size < 2:
            kernel_size = 2
        dropout = trial.suggest_float("dropout", float(hpo.get("dropout", [0.0, 0.35])[0]), float(hpo.get("dropout", [0.0, 0.35])[1]))
        lr = trial.suggest_float("lr", float(hpo.get("lr", [1e-4, 3e-3])[0]), float(hpo.get("lr", [1e-4, 3e-3])[1]), log=True)
        weight_decay = trial.suggest_float(
            "weight_decay",
            float(hpo.get("weight_decay", [1e-8, 1e-3])[0]),
            float(hpo.get("weight_decay", [1e-8, 1e-3])[1]),
            log=True,
        )
        batch_choices = [int(v) for v in hpo.get("batch_size_choices", [64, 128, 256])]
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
        max_epochs = trial.suggest_int(
            "max_epochs",
            int(hpo.get("max_epochs", [8, 24])[0]),
            int(hpo.get("max_epochs", [8, 24])[1]),
        )
        patience = trial.suggest_int(
            "patience",
            int(hpo.get("patience", [3, 7])[0]),
            int(hpo.get("patience", [3, 7])[1]),
        )
        return {
            "lookback": int(lookback),
            "levels": int(levels),
            "channels": int(channels),
            "kernel_size": int(kernel_size),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
        }

    def _fit_fold(
        self,
        *,
        X_all: np.ndarray,
        y_all: np.ndarray,
        train_end: np.ndarray,
        val_end: np.ndarray,
        params: Dict[str, Any],
        seed_offset: int = 0,
        heartbeat_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        if len(train_end) < 64 or len(val_end) < 64:
            return {"score": 0.0, "ap": 0.0, "auc": 0.5, "state_dict": None, "epochs_ran": 0}

        _set_global_seed(int(self.cfg.get("random_seed", 42)) + int(seed_offset))
        device = torch.device(str(self.cfg.get("device", "cpu")).lower())

        channels = [int(params["channels"]) for _ in range(int(params["levels"]))]
        model_params = {
            "input_dim": int(X_all.shape[1]),
            "channels": channels,
            "kernel_size": int(params["kernel_size"]),
            "dropout": float(params["dropout"]),
        }
        model = TCNBinaryClassifier(**model_params).to(device)

        y_train = y_all[train_end]
        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        pos_weight = float(n_neg / max(n_pos, 1.0))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))

        train_ds = SequenceIndexDataset(X_all, y_all, train_end, int(params["lookback"]))
        val_ds = SequenceIndexDataset(X_all, y_all, val_end, int(params["lookback"]))
        train_loader = DataLoader(train_ds, batch_size=int(params["batch_size"]), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]) * 2, shuffle=False)

        best_score = -np.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_ap = 0.0
        best_auc = 0.5
        best_epoch = 0
        stale = 0
        fit_started = time.perf_counter()
        heartbeat_every = max(int(self.cfg.get("epoch_heartbeat_seconds", 120)), 0)
        last_heartbeat = fit_started

        for epoch in range(1, int(params["max_epochs"]) + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += float(loss.item())
                epoch_batches += 1

            model.eval()
            val_probs: List[np.ndarray] = []
            val_targets: List[np.ndarray] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    p = torch.sigmoid(logits).detach().cpu().numpy()
                    val_probs.append(p)
                    val_targets.append(yb.numpy())
            if not val_probs:
                break
            p_vec = np.concatenate(val_probs)
            y_vec = np.concatenate(val_targets).astype(int)
            ap, auc, score = self._binary_scores(y_vec, p_vec)
            if score > best_score:
                best_score = score
                best_ap = ap
                best_auc = auc
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(params["patience"]):
                    break

            now = time.perf_counter()
            if heartbeat_every > 0 and (now - last_heartbeat) >= heartbeat_every:
                elapsed = now - fit_started
                epoch_done = max(epoch, 1)
                max_epochs = max(int(params["max_epochs"]), 1)
                eta = (elapsed / epoch_done) * max(max_epochs - epoch_done, 0)
                avg_loss = epoch_loss_sum / max(epoch_batches, 1)
                prefix = f"{heartbeat_label} " if heartbeat_label else ""
                console_stage(
                    f"{self.target} TCN epoch heartbeat",
                    (
                        f"{prefix}epoch={epoch_done}/{max_epochs} "
                        f"avg_loss={avg_loss:.5f} best_score={best_score:.4f} "
                        f"elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
                    ),
                    status="info",
                )
                last_heartbeat = now

        return {
            "score": float(best_score if np.isfinite(best_score) else 0.0),
            "ap": float(best_ap),
            "auc": float(best_auc),
            "state_dict": best_state,
            "model_params": model_params,
            "epochs_ran": int(best_epoch),
        }

    def _evaluate_cv(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        params: Dict[str, Any],
        *,
        trial: Optional[optuna.Trial] = None,
    ) -> Dict[str, Any]:
        cv_splits = int(self.cfg.get("cv_splits", 4))
        embargo = int(self.cfg.get("embargo_bars", 0))
        tscv = TimeSeriesSplit(n_splits=max(cv_splits, 2))
        cv_started = time.perf_counter()
        fold_results: List[Dict[str, Any]] = []
        transform_dim = 0
        splits = list(tscv.split(X_df))
        total_folds = max(len(splits), 1)
        trial_name = "final_eval"
        if trial is not None:
            trial_name = f"trial={trial.number + 1}"

        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            pre = self._build_preprocessor(X_df)
            pre.fit(X_df.iloc[tr_idx])
            X_all = self._ensure_dense_float32(pre.transform(X_df))
            transform_dim = int(X_all.shape[1])

            max_train_end = int(va_idx.min()) - max(embargo, 0) - 1
            train_end = self._build_end_indices(tr_idx, int(params["lookback"]), max_train_end=max_train_end)
            val_end = self._build_end_indices(va_idx, int(params["lookback"]))

            fold = self._fit_fold(
                X_all=X_all,
                y_all=y,
                train_end=train_end,
                val_end=val_end,
                params=params,
                seed_offset=fold_idx,
                heartbeat_label=f"{trial_name} fold={fold_idx}/{total_folds}",
            )
            fold_results.append(
                {
                    "fold": fold_idx,
                    "score": float(fold["score"]),
                    "ap": float(fold["ap"]),
                    "auc": float(fold["auc"]),
                    "epochs_ran": int(fold["epochs_ran"]),
                }
            )

            if bool(self.cfg.get("fold_heartbeat", True)):
                elapsed = time.perf_counter() - cv_started
                rate = fold_idx / max(elapsed, 1e-6)
                eta = (total_folds - fold_idx) / max(rate, 1e-6)
                partial = float(np.mean([row["score"] for row in fold_results]))
                console_stage(
                    f"{self.target} TCN fold progress",
                    (
                        f"{trial_name} fold={fold_idx}/{total_folds} "
                        f"partial_score={partial:.4f} elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
                    ),
                    status="info",
                )

            if trial is not None:
                partial = float(np.mean([row["score"] for row in fold_results]))
                trial.report(partial, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not fold_results:
            return {
                "score_mean": 0.0,
                "ap_mean": 0.0,
                "auc_mean": 0.5,
                "score_std": 0.0,
                "folds": [],
                "transform_dim": transform_dim,
            }
        score_arr = np.array([row["score"] for row in fold_results], dtype=float)
        ap_arr = np.array([row["ap"] for row in fold_results], dtype=float)
        auc_arr = np.array([row["auc"] for row in fold_results], dtype=float)
        return {
            "score_mean": float(np.mean(score_arr)),
            "score_std": float(np.std(score_arr)),
            "ap_mean": float(np.mean(ap_arr)),
            "auc_mean": float(np.mean(auc_arr)),
            "folds": fold_results,
            "transform_dim": int(transform_dim),
        }

    def _fit_final_model(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        n_rows = len(X_df)
        holdout_frac = float(self.cfg.get("holdout_frac", 0.15))
        holdout = max(int(n_rows * holdout_frac), int(params["lookback"]) * 2, 512)
        holdout = min(holdout, max(1024, n_rows // 3))
        split = max(n_rows - holdout, int(params["lookback"]) + 64)

        fit_idx = np.arange(0, split, dtype=int)
        cal_idx = np.arange(split, n_rows, dtype=int)

        pre = self._build_preprocessor(X_df)
        pre.fit(X_df.iloc[fit_idx])
        X_all = self._ensure_dense_float32(pre.transform(X_df))

        train_end = self._build_end_indices(fit_idx, int(params["lookback"]))
        val_end = self._build_end_indices(cal_idx, int(params["lookback"]))
        fold = self._fit_fold(
            X_all=X_all,
            y_all=y,
            train_end=train_end,
            val_end=val_end,
            params=params,
            seed_offset=999,
            heartbeat_label="final_fit",
        )
        if fold["state_dict"] is None:
            raise RuntimeError("TCN final fit failed: no valid training state produced.")

        model_params = fold["model_params"]
        raw_model = TCNBinaryClassifier(**model_params)
        raw_model.load_state_dict(fold["state_dict"], strict=True)
        raw_model.eval()

        # Calibration
        calibrator = None
        cal_method = str(self.cfg.get("calibrator", "auto")).lower()
        calibration_report: Dict[str, Any] = {
            "method": "none",
            "brier_raw": None,
            "brier_calibrated": None,
            "holdout_rows": int(len(val_end)),
            "holdout_eval": None,
        }
        threshold_report = {
            "metric": str(deepcopy(self.cfg.get("threshold_tuning", {})).get("metric", "f1")),
            "threshold": float(deepcopy(self.cfg.get("threshold_tuning", {})).get("default_threshold", 0.5)),
            "best_value": None,
            "constraints": {},
            "metrics": None,
        }

        p_raw = np.array([], dtype=float)
        y_cal = np.array([], dtype=int)
        if len(val_end) >= 64:
            val_ds = SequenceIndexDataset(X_all, y, val_end, int(params["lookback"]))
            val_loader = DataLoader(val_ds, batch_size=int(params["batch_size"]) * 2, shuffle=False)
            raw_probs: List[np.ndarray] = []
            raw_targets: List[np.ndarray] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = raw_model(xb)
                    p = torch.sigmoid(logits).cpu().numpy()
                    raw_probs.append(p)
                    raw_targets.append(yb.numpy())
            p_raw = np.concatenate(raw_probs) if raw_probs else np.array([], dtype=float)
            y_cal = np.concatenate(raw_targets).astype(int) if raw_targets else np.array([], dtype=int)

        p_for_decision = p_raw
        if cal_method != "none" and len(p_raw) > 0 and np.unique(y_cal).size > 1 and np.unique(p_raw).size > 1:
            method_for_emp = "auto" if cal_method == "auto" else ("platt" if cal_method == "platt" else ("isotonic" if cal_method == "isotonic" else "auto"))
            if cal_method == "empirical":
                method_for_emp = "auto"
            calibrator = EmpiricalCalibrator(method=method_for_emp).calibrate(p_raw, y_cal)

            if hasattr(calibrator, "predict_proba"):
                p_cal = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
                chosen = "platt"
            elif isinstance(calibrator, IsotonicRegression):
                p_cal = calibrator.predict(p_raw)
                chosen = "isotonic"
            elif hasattr(calibrator, "predict"):
                p_cal = calibrator.predict(p_raw)
                chosen = calibrator.__class__.__name__
            else:
                p_cal = np.asarray([float(calibrator(v)) for v in p_raw], dtype=float)
                chosen = "callable"

            p_for_decision = np.asarray(p_cal, dtype=float)
            calibration_report = {
                "method": chosen,
                "brier_raw": float(brier_score_loss(y_cal, p_raw)),
                "brier_calibrated": float(brier_score_loss(y_cal, p_for_decision)),
                "holdout_rows": int(len(y_cal)),
                "holdout_eval": None,
            }

        if len(p_for_decision) > 0 and len(y_cal) == len(p_for_decision):
            threshold_report = self._tune_threshold(y_cal, p_for_decision)
            calibration_report["holdout_eval"] = self._evaluate_binary_predictions(
                y_cal,
                p_for_decision,
                float(threshold_report["threshold"]),
            )

        wrapped = TCNCalibratedModel(
            feature_cols=list(X_df.columns),
            lookback=int(params["lookback"]),
            preprocessor=pre,
            model_params=model_params,
            state_dict=fold["state_dict"],
            calibrator=calibrator,
            decision_threshold=float(threshold_report["threshold"]),
        )
        return {
            "model": wrapped,
            "model_params": model_params,
            "fold_metrics": {
                "score": float(fold["score"]),
                "ap": float(fold["ap"]),
                "auc": float(fold["auc"]),
                "epochs_ran": int(fold["epochs_ran"]),
            },
            "calibration": calibration_report,
            "threshold_tuning": threshold_report,
            "transform_dim": int(X_all.shape[1]),
            "lookback": int(params["lookback"]),
        }

    def _compute_acceptance_report(
        self,
        *,
        final_fit: Dict[str, Any],
        X_dev: pd.DataFrame,
        X_accept: Optional[pd.DataFrame],
        y_accept: Optional[np.ndarray],
        acceptance_rows: int,
        cv_score: float,
    ) -> Dict[str, Any]:
        acceptance_cfg = deepcopy(self.cfg.get("acceptance", {}))
        acceptance_report: Dict[str, Any] = {
            "enabled": bool(acceptance_cfg.get("enabled", True)),
            "rows": int(acceptance_rows),
            "metrics": None,
            "gate": {"pass": None, "reasons": []},
        }
        if X_accept is None or y_accept is None or len(X_accept) <= 0:
            return acceptance_report

        lookback = int(final_fit.get("lookback", 1))
        context_rows = max(lookback - 1, 0)
        X_context = X_dev.tail(context_rows) if context_rows > 0 else X_dev.iloc[0:0]
        X_eval = pd.concat([X_context, X_accept], axis=0)
        p_eval_full = final_fit["model"].predict_proba(X_eval)[:, 1]
        p_accept = np.asarray(p_eval_full[-len(X_accept):], dtype=float)
        decision_thr = float(final_fit.get("threshold_tuning", {}).get("threshold", 0.5))
        acceptance_metrics = self._evaluate_binary_predictions(y_accept, p_accept, decision_thr)
        acceptance_report["metrics"] = acceptance_metrics

        reasons: List[str] = []
        max_drop = float(acceptance_cfg.get("max_score_drop", 0.05))
        min_score = acceptance_cfg.get("min_score", None)
        min_precision = float(acceptance_cfg.get("min_precision_at_threshold", 0.10))
        if min_score is not None and float(acceptance_metrics["score"]) < float(min_score):
            reasons.append("acceptance_score_below_min")
        if (float(cv_score) - float(acceptance_metrics["score"])) > max_drop:
            reasons.append("acceptance_drop_exceeds_tolerance")
        if float(acceptance_metrics["precision_at_threshold"]) < min_precision:
            reasons.append("acceptance_precision_below_min")
        acceptance_report["gate"] = {
            "pass": len(reasons) == 0,
            "reasons": reasons,
            "max_score_drop": max_drop,
            "min_score": min_score,
            "min_precision_at_threshold": min_precision,
        }
        return acceptance_report

    def _read_tree_baseline(self) -> Optional[float]:
        tree_baseline = None
        if self.tree_manifest_path is not None and self.tree_manifest_path.exists():
            try:
                payload = json.loads(self.tree_manifest_path.read_text(encoding="utf-8"))
                tree_baseline = (
                    payload.get("metrics", {})
                    .get("by_model", {})
                    .get(self.target, {})
                    .get("cv_score")
                )
                if tree_baseline is not None:
                    tree_baseline = float(tree_baseline)
            except Exception:
                tree_baseline = None
        return tree_baseline

    @staticmethod
    def _best_study_value(study: optuna.Study) -> Optional[float]:
        try:
            return float(study.best_value)
        except Exception:
            return None

    def _build_interrupt_checkpoint_result(
        self,
        *,
        study: optuna.Study,
        X_dev: pd.DataFrame,
        y_dev: np.ndarray,
        X_accept: Optional[pd.DataFrame],
        y_accept: Optional[np.ndarray],
        acceptance_rows: int,
        feature_cols: List[str],
        class_counts: Dict[Any, Any],
        started: float,
        run_started_epoch: float,
        n_trials: int,
        study_name: str,
        storage_uri: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            self._write_progress_snapshot(
                {
                    "status": "interrupted",
                    "phase": "hpo",
                    "asset": self.asset,
                    "target": self.target,
                    "run_started_at_epoch": float(run_started_epoch),
                    "requested_trials": int(n_trials),
                    "completed_trials": int(self._count_completed_trials(study)),
                    "remaining_trials": int(max(n_trials - self._count_completed_trials(study), 0)),
                    "elapsed_sec": float(time.time() - float(run_started_epoch)),
                    "study_name": study_name,
                    "hpo_storage": storage_uri,
                    "checkpoint_saved": False,
                    "checkpoint_reason": reason,
                },
                append_event=True,
            )
            raise KeyboardInterrupt

        console_stage(
            f"{self.target} TCN checkpoint",
            "interrupt received | finalizing best completed trial",
            status="warn",
        )
        best_params = deepcopy(study.best_params)
        best_value = self._best_study_value(study)
        if best_value is None:
            best_value = float("nan")
        final_fit = self._fit_final_model(X_dev, y_dev, best_params)
        acceptance_report = self._compute_acceptance_report(
            final_fit=final_fit,
            X_dev=X_dev,
            X_accept=X_accept,
            y_accept=y_accept,
            acceptance_rows=acceptance_rows,
            cv_score=float(best_value),
        )
        tree_baseline = self._read_tree_baseline()
        metrics = {
            "cv_score": float(best_value),
            "cv_ap": float(best_value) if np.isfinite(best_value) else None,
            "cv_auc": None,
            "cv_score_std": None,
            "best_params": best_params,
            "hpo_trials": int(n_trials),
            "hpo_trials_completed": int(self._count_completed_trials(study)),
            "hpo_study_name": study.study_name,
            "hpo_storage": storage_uri,
            "class_counts": class_counts,
            "dev_rows": int(len(X_dev)),
            "acceptance_rows": int(acceptance_rows),
            "selected_feature_cols": feature_cols,
            "feature_transform_dim": int(final_fit["transform_dim"]),
            "folds": [],
            "calibration": final_fit["calibration"],
            "threshold_tuning": final_fit["threshold_tuning"],
            "stability": {
                "enabled": bool(deepcopy(self.cfg.get("stability", {})).get("enabled", True)),
                "rows": [],
                "gate": {"pass": None},
                "skipped_due_to_interrupt": True,
            },
            "acceptance": acceptance_report,
            "feature_dominance_audit": {
                "skipped_due_to_interrupt": True,
            },
            "tree_baseline_cv_score": tree_baseline,
            "delta_vs_tree_cv_score": (
                float(best_value) - tree_baseline
                if tree_baseline is not None and np.isfinite(best_value)
                else None
            ),
            "tcn_fit_runtime_sec": float(time.perf_counter() - started),
            "checkpoint_interrupted": True,
            "checkpoint_reason": reason,
            "metrics_partial": True,
            "cv_metric_source": "optuna_best_trial",
        }
        self._write_progress_snapshot(
            {
                "status": "checkpoint_saved",
                "phase": "interrupted_checkpoint",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "elapsed_sec": float(time.time() - float(run_started_epoch)),
                "requested_trials": int(n_trials),
                "completed_trials": int(self._count_completed_trials(study)),
                "remaining_trials": int(max(n_trials - self._count_completed_trials(study), 0)),
                "cv_score": float(metrics["cv_score"]) if metrics.get("cv_score") is not None else None,
                "tree_baseline_cv_score": metrics.get("tree_baseline_cv_score"),
                "delta_vs_tree_cv_score": metrics.get("delta_vs_tree_cv_score"),
                "acceptance_gate": (metrics.get("acceptance", {}) or {}).get("gate"),
                "stability_gate": (metrics.get("stability", {}) or {}).get("gate"),
                "study_name": study_name,
                "hpo_storage": storage_uri,
                "best_params": best_params,
                "threshold": ((metrics.get("threshold_tuning", {}) or {}).get("threshold")),
                "checkpoint_saved": True,
                "checkpoint_reason": reason,
                "metrics_partial": True,
            },
            append_event=True,
        )
        console_stage(
            f"{self.target} TCN checkpoint saved",
            (
                f"cv_score={metrics['cv_score']:.4f} "
                f"completed_trials={metrics['hpo_trials_completed']}/{metrics['hpo_trials']}"
            ),
            status="warn",
        )
        return {
            "model": final_fit["model"],
            "metrics": metrics,
            "feature_cols": feature_cols,
            "runtime_sec": float(time.perf_counter() - started),
            "outcome": "checkpoint_saved",
        }

    def train(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        started = time.perf_counter()
        if self.label_col not in train_df.columns:
            raise KeyError(f"Missing target column for {self.target}: {self.label_col}")

        feature_cols = self._resolve_feature_cols(train_df)
        if len(feature_cols) < 8:
            raise ValueError(f"Too few features for TCN training ({len(feature_cols)}).")

        X_df = train_df[feature_cols].copy()
        y = train_df[self.label_col].astype(int).values
        class_counts = pd.Series(y).value_counts().to_dict()
        X_dev, y_dev, X_accept, y_accept, acceptance_rows = self._split_dev_acceptance(X_df, y)
        console_stage(
            f"{self.target} TCN setup",
            (
                f"rows={fmt_num(len(X_df))} dev_rows={fmt_num(len(X_dev))} "
                f"accept_rows={fmt_num(acceptance_rows)} features={fmt_num(len(feature_cols))} "
                f"class_counts={class_counts}"
            ),
            status="info",
        )

        n_trials = int(self.cfg.get("hpo_trials", 20))
        sampler = optuna.samplers.TPESampler(seed=int(self.cfg.get("random_seed", 42)))
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(int(self.cfg.get("hpo_pruner_warmup_steps", 1)), 0))
        storage_uri = self.cfg.get("hpo_storage")
        if storage_uri is not None:
            storage_uri = str(storage_uri).strip() or None
        study_name = str(self.cfg.get("hpo_study_name") or f"{self.asset}_{self.target}_tcn")
        hpo_resume = bool(self.cfg.get("hpo_resume", True))
        run_started_epoch = self._read_snapshot_started_at() if (storage_uri and hpo_resume) else None
        if run_started_epoch is None:
            run_started_epoch = time.time()
        if storage_uri:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage=storage_uri,
                load_if_exists=hpo_resume,
            )
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)
            cv = self._evaluate_cv(X_dev, y_dev, params, trial=trial)
            return float(cv["score_mean"])

        completed_before = self._count_completed_trials(study)
        if storage_uri:
            trials_to_run = max(int(n_trials) - int(completed_before), 0)
        else:
            trials_to_run = max(int(n_trials), 1)
        self._write_progress_snapshot(
            {
                "status": "running",
                "phase": "hpo",
                "asset": self.asset,
                "target": self.target,
                "rows": int(len(X_df)),
                "dev_rows": int(len(X_dev)),
                "acceptance_rows": int(acceptance_rows),
                "feature_count": int(len(feature_cols)),
                "run_started_at_epoch": float(run_started_epoch),
                "requested_trials": int(n_trials),
                "completed_trials": int(completed_before),
                "remaining_trials": int(max(n_trials - completed_before, 0)),
                "cv_splits": int(self.cfg.get("cv_splits", 4)),
                "study_name": study_name,
                "hpo_storage": storage_uri,
            },
            append_event=True,
        )

        console_stage(
            f"{self.target} TCN HPO",
            (
                f"requested={n_trials} run_now={trials_to_run} done_before={completed_before} "
                f"cv_splits={int(self.cfg.get('cv_splits', 4))}"
            ),
            status="info",
        )
        log_every = max(int(self.cfg.get("hpo_log_every_n_trials", 1)), 1)
        adaptive_stop_enabled = bool(self.cfg.get("hpo_adaptive_stop", False))
        adaptive_min_completed = max(int(self.cfg.get("hpo_adaptive_min_completed_trials", 12)), 1)
        adaptive_no_improve = max(int(self.cfg.get("hpo_adaptive_no_improve_trials", 8)), 1)
        adaptive_min_delta = float(self.cfg.get("hpo_adaptive_min_delta", 0.001))
        best_seen, done_at_last_improve = self._scan_best_state(study)
        stop_triggered = False
        stop_reason = None

        def _hpo_callback(study_ref: optuna.Study, trial_ref: optuna.Trial) -> None:
            nonlocal best_seen, done_at_last_improve, stop_triggered, stop_reason
            done = self._count_completed_trials(study_ref)
            current_best = best_seen
            try:
                if study_ref.best_trial is not None and study_ref.best_trial.value is not None:
                    current_best = float(study_ref.best_trial.value)
            except Exception:
                current_best = best_seen

            if current_best is not None:
                if best_seen is None or current_best >= (best_seen + adaptive_min_delta):
                    best_seen = current_best
                    done_at_last_improve = done

            if adaptive_stop_enabled:
                stale_trials = int(max(done - done_at_last_improve, 0))
                if done >= adaptive_min_completed and stale_trials >= adaptive_no_improve:
                    stop_triggered = True
                    stop_reason = (
                        f"adaptive_plateau(no_improve={stale_trials} trials, "
                        f"min_delta={adaptive_min_delta}, min_completed={adaptive_min_completed})"
                    )
                    console_stage(
                        f"{self.target} TCN HPO",
                        f"early stop triggered | {stop_reason}",
                        status="warn",
                    )
                    self._write_progress_snapshot(
                        {
                            "status": "running",
                            "phase": "hpo",
                            "asset": self.asset,
                            "target": self.target,
                            "rows": int(len(X_df)),
                            "dev_rows": int(len(X_dev)),
                            "acceptance_rows": int(acceptance_rows),
                            "feature_count": int(len(feature_cols)),
                            "run_started_at_epoch": float(run_started_epoch),
                            "requested_trials": int(n_trials),
                            "completed_trials": int(done),
                            "remaining_trials": int(max(n_trials - done, 0)),
                            "best_value": float(best_seen) if best_seen is not None else None,
                            "last_trial_number": int(trial_ref.number + 1),
                            "last_trial_value": float(trial_ref.value) if trial_ref.value is not None else None,
                            "elapsed_sec": float(time.time() - float(run_started_epoch)),
                            "cv_splits": int(self.cfg.get("cv_splits", 4)),
                            "study_name": study_ref.study_name,
                            "hpo_storage": storage_uri,
                            "hpo_adaptive_stop": True,
                            "hpo_adaptive_no_improve_trials": int(adaptive_no_improve),
                            "hpo_adaptive_min_completed_trials": int(adaptive_min_completed),
                            "hpo_adaptive_min_delta": float(adaptive_min_delta),
                            "hpo_stop_triggered": True,
                            "hpo_stop_reason": stop_reason,
                        },
                        append_event=True,
                    )
                    study_ref.stop()
                    return

            if done % log_every != 0:
                return
            elapsed = time.time() - float(run_started_epoch)
            rate = done / max(elapsed, 1e-6)
            eta = max(n_trials - done, 0) / max(rate, 1e-6)
            durations = [
                float(t.duration.total_seconds())
                for t in study_ref.trials
                if t.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL} and t.duration is not None
            ]
            avg_trial_sec = float(np.mean(durations)) if durations else None
            eta_by_avg = (max(n_trials - done, 0) * avg_trial_sec) if avg_trial_sec is not None else None
            try:
                best = float(study_ref.best_value)
            except Exception:
                best = float("nan")
            trial_value = None
            try:
                if trial_ref.value is not None:
                    trial_value = float(trial_ref.value)
            except Exception:
                trial_value = None
            console_stage(
                f"{self.target} TCN HPO progress",
                (
                    f"trials={done}/{n_trials} remaining={max(n_trials - done, 0)} "
                    f"best={best:.4f} trial={trial_ref.number + 1} "
                    f"elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(eta)} "
                    f"eta_avg={fmt_seconds(eta_by_avg) if eta_by_avg is not None else '-'}"
                ),
                status="info",
            )
            self._write_progress_snapshot(
                {
                    "status": "running",
                    "phase": "hpo",
                    "asset": self.asset,
                    "target": self.target,
                    "rows": int(len(X_df)),
                    "dev_rows": int(len(X_dev)),
                    "acceptance_rows": int(acceptance_rows),
                    "feature_count": int(len(feature_cols)),
                    "run_started_at_epoch": float(run_started_epoch),
                    "requested_trials": int(n_trials),
                    "completed_trials": int(done),
                    "remaining_trials": int(max(n_trials - done, 0)),
                    "best_value": best,
                    "last_trial_number": int(trial_ref.number + 1),
                    "last_trial_value": trial_value,
                    "elapsed_sec": float(elapsed),
                    "eta_sec": float(eta),
                    "avg_trial_sec": avg_trial_sec,
                    "eta_by_avg_sec": float(eta_by_avg) if eta_by_avg is not None else None,
                    "cv_splits": int(self.cfg.get("cv_splits", 4)),
                    "study_name": study_ref.study_name,
                    "hpo_storage": storage_uri,
                    "hpo_adaptive_stop": bool(adaptive_stop_enabled),
                    "hpo_adaptive_no_improve_trials": int(adaptive_no_improve),
                    "hpo_adaptive_min_completed_trials": int(adaptive_min_completed),
                    "hpo_adaptive_min_delta": float(adaptive_min_delta),
                    "hpo_stop_triggered": bool(stop_triggered),
                    "hpo_stop_reason": stop_reason,
                },
                append_event=True,
            )

        try:
            if trials_to_run > 0:
                study.optimize(
                    objective,
                    n_trials=trials_to_run,
                    show_progress_bar=False,
                    gc_after_trial=bool(self.cfg.get("hpo_gc_after_trial", True)),
                    callbacks=[_hpo_callback],
                )
            else:
                console_stage(
                    f"{self.target} TCN HPO",
                    "resume hit: requested trials already completed, skipping optimization",
                    status="ok",
                )
        except KeyboardInterrupt:
            return self._build_interrupt_checkpoint_result(
                study=study,
                X_dev=X_dev,
                y_dev=y_dev,
                X_accept=X_accept,
                y_accept=y_accept,
                acceptance_rows=acceptance_rows,
                feature_cols=feature_cols,
                class_counts=class_counts,
                started=started,
                run_started_epoch=float(run_started_epoch),
                n_trials=n_trials,
                study_name=study_name,
                storage_uri=storage_uri,
                reason="keyboard_interrupt",
            )
        done_after_hpo = self._count_completed_trials(study)
        try:
            best_after_hpo = float(study.best_value)
        except Exception:
            best_after_hpo = None
        elapsed_after_hpo = time.time() - float(run_started_epoch)
        self._write_progress_snapshot(
            {
                "status": "running",
                "phase": "hpo_done",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "requested_trials": int(n_trials),
                "completed_trials": int(done_after_hpo),
                "remaining_trials": int(max(n_trials - done_after_hpo, 0)),
                "best_value": best_after_hpo,
                "elapsed_sec": float(elapsed_after_hpo),
                "cv_splits": int(self.cfg.get("cv_splits", 4)),
                "study_name": study_name,
                "hpo_storage": storage_uri,
                "hpo_adaptive_stop": bool(adaptive_stop_enabled),
                "hpo_adaptive_no_improve_trials": int(adaptive_no_improve),
                "hpo_adaptive_min_completed_trials": int(adaptive_min_completed),
                "hpo_adaptive_min_delta": float(adaptive_min_delta),
                "hpo_stop_triggered": bool(stop_triggered),
                "hpo_stop_reason": stop_reason,
            },
            append_event=True,
        )

        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            raise RuntimeError("TCN HPO produced no completed trials.")
        best_params = deepcopy(study.best_params)

        cv_best = self._evaluate_cv(X_dev, y_dev, best_params, trial=None)
        self._write_progress_snapshot(
            {
                "status": "running",
                "phase": "cv_best",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "requested_trials": int(n_trials),
                "completed_trials": int(self._count_completed_trials(study)),
                "remaining_trials": int(max(n_trials - self._count_completed_trials(study), 0)),
                "best_value": float(cv_best["score_mean"]),
                "cv_score": float(cv_best["score_mean"]),
                "cv_ap": float(cv_best["ap_mean"]),
                "cv_auc": float(cv_best["auc_mean"]),
                "elapsed_sec": float(time.time() - float(run_started_epoch)),
                "study_name": study_name,
                "hpo_storage": storage_uri,
            },
            append_event=True,
        )

        stability_cfg = deepcopy(self.cfg.get("stability", {}))
        stability_rows: List[Dict[str, Any]] = []
        if bool(stability_cfg.get("enabled", True)):
            seeds_raw = stability_cfg.get("seeds", [self.cfg.get("random_seed", 42)])
            seeds: List[int] = []
            for seed in seeds_raw if isinstance(seeds_raw, (list, tuple)) else [seeds_raw]:
                try:
                    seeds.append(int(seed))
                except Exception:
                    continue
            seeds = list(dict.fromkeys(seeds)) or [int(self.cfg.get("random_seed", 42))]
            original_seed = int(self.cfg.get("random_seed", 42))
            for seed in seeds:
                self.cfg["random_seed"] = int(seed)
                self._set_runtime_knobs()
                cv_seed = self._evaluate_cv(X_dev, y_dev, best_params, trial=None)
                stability_rows.append(
                    {
                        "seed": int(seed),
                        "cv_score": float(cv_seed["score_mean"]),
                        "cv_ap": float(cv_seed["ap_mean"]),
                        "cv_auc": float(cv_seed["auc_mean"]),
                    }
                )
            self.cfg["random_seed"] = original_seed
            self._set_runtime_knobs()

        if stability_rows:
            stab_scores = np.array([row["cv_score"] for row in stability_rows], dtype=float)
            max_std = float(stability_cfg.get("max_std", 0.02))
            max_drop = float(stability_cfg.get("max_drop_vs_best", 0.04))
            stability_pass = bool(
                np.std(stab_scores) <= max_std and (float(cv_best["score_mean"]) - np.min(stab_scores)) <= max_drop
            )
            stability_report = {
                "enabled": True,
                "rows": stability_rows,
                "score_mean": float(np.mean(stab_scores)),
                "score_std": float(np.std(stab_scores)),
                "score_min": float(np.min(stab_scores)),
                "gate": {
                    "pass": stability_pass,
                    "max_std": max_std,
                    "max_drop_vs_best": max_drop,
                },
            }
        else:
            stability_report = {"enabled": bool(stability_cfg.get("enabled", True)), "rows": [], "gate": {"pass": None}}
        self._write_progress_snapshot(
            {
                "status": "running",
                "phase": "stability_done",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "cv_score": float(cv_best["score_mean"]),
                "stability_gate": (stability_report.get("gate") or {}),
                "elapsed_sec": float(time.time() - float(run_started_epoch)),
                "study_name": study_name,
                "hpo_storage": storage_uri,
            },
            append_event=True,
        )

        final_fit = self._fit_final_model(X_dev, y_dev, best_params)
        self._write_progress_snapshot(
            {
                "status": "running",
                "phase": "final_fit_done",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "cv_score": float(cv_best["score_mean"]),
                "threshold": float((final_fit.get("threshold_tuning") or {}).get("threshold", 0.5)),
                "elapsed_sec": float(time.time() - float(run_started_epoch)),
                "study_name": study_name,
                "hpo_storage": storage_uri,
            },
            append_event=True,
        )

        acceptance_report = self._compute_acceptance_report(
            final_fit=final_fit,
            X_dev=X_dev,
            X_accept=X_accept,
            y_accept=y_accept,
            acceptance_rows=acceptance_rows,
            cv_score=float(cv_best["score_mean"]),
        )

        dominance_report = self._feature_dominance_audit(final_fit["model"], X_dev, y_dev)

        tree_baseline = self._read_tree_baseline()

        metrics = {
            "cv_score": float(cv_best["score_mean"]),
            "cv_ap": float(cv_best["ap_mean"]),
            "cv_auc": float(cv_best["auc_mean"]),
            "cv_score_std": float(cv_best["score_std"]),
            "best_params": best_params,
            "hpo_trials": int(n_trials),
            "hpo_trials_completed": int(self._count_completed_trials(study)),
            "hpo_study_name": study.study_name,
            "hpo_storage": storage_uri,
            "class_counts": class_counts,
            "dev_rows": int(len(X_dev)),
            "acceptance_rows": int(acceptance_rows),
            "selected_feature_cols": feature_cols,
            "feature_transform_dim": int(cv_best["transform_dim"] or final_fit["transform_dim"]),
            "folds": cv_best["folds"],
            "calibration": final_fit["calibration"],
            "threshold_tuning": final_fit["threshold_tuning"],
            "stability": stability_report,
            "acceptance": acceptance_report,
            "feature_dominance_audit": dominance_report,
            "tree_baseline_cv_score": tree_baseline,
            "delta_vs_tree_cv_score": (
                float(cv_best["score_mean"]) - tree_baseline if tree_baseline is not None else None
            ),
            "tcn_fit_runtime_sec": float(time.perf_counter() - started),
        }
        self._write_progress_snapshot(
            {
                "status": "completed",
                "phase": "done",
                "asset": self.asset,
                "target": self.target,
                "run_started_at_epoch": float(run_started_epoch),
                "elapsed_sec": float(time.time() - float(run_started_epoch)),
                "requested_trials": int(n_trials),
                "completed_trials": int(self._count_completed_trials(study)),
                "remaining_trials": 0,
                "cv_score": float(metrics["cv_score"]),
                "cv_ap": float(metrics["cv_ap"]),
                "cv_auc": float(metrics["cv_auc"]),
                "tree_baseline_cv_score": metrics.get("tree_baseline_cv_score"),
                "delta_vs_tree_cv_score": metrics.get("delta_vs_tree_cv_score"),
                "acceptance_gate": (metrics.get("acceptance", {}) or {}).get("gate"),
                "stability_gate": (metrics.get("stability", {}) or {}).get("gate"),
                "dominance_audit": (metrics.get("feature_dominance_audit") or {}),
                "study_name": study_name,
                "hpo_storage": storage_uri,
                "best_params": best_params,
                "threshold": ((metrics.get("threshold_tuning", {}) or {}).get("threshold")),
            },
            append_event=True,
        )

        console_stage(
            f"{self.target} TCN done",
            (
                f"cv_score={metrics['cv_score']:.4f} "
                f"cv_ap={metrics['cv_ap']:.4f} "
                f"accept_score={((metrics['acceptance'].get('metrics') or {}).get('score'))} "
                f"delta_vs_tree={metrics['delta_vs_tree_cv_score']}"
            ),
            status="ok",
        )

        return {
            "model": final_fit["model"],
            "metrics": metrics,
            "feature_cols": feature_cols,
            "runtime_sec": float(time.perf_counter() - started),
            "outcome": "trained",
        }
