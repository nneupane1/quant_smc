"""
Empirical probability calibration utilities.
"""

import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from quant_system.utils.logger import log


class HistogramBinningCalibrator:
    """
    Pickle-friendly histogram-binning calibrator.
    """

    def __init__(self, edges: np.ndarray, bin_means: Dict[int, float]):
        self.edges = np.asarray(edges, dtype=float)
        self.bin_means = {int(k): float(v) for k, v in bin_means.items()}

    def predict(self, p: np.ndarray) -> np.ndarray:
        vals = np.asarray(p, dtype=float).reshape(-1)
        idx = np.digitize(vals, self.edges) - 1
        idx = np.clip(idx, 0, len(self.edges) - 2)
        return np.array([self.bin_means.get(int(i), 0.5) for i in idx], dtype=float)


class EmpiricalCalibrator:
    """
    Provides Platt, Isotonic, and Histogram Binning calibration.
    """

    def __init__(self, method: str = "auto"):
        self.method = method
        log(f"EmpiricalCalibrator (method={method}) ready.")

    def _platt(self, p: np.ndarray, y: np.ndarray):
        model = LogisticRegression(max_iter=500).fit(p.reshape(-1, 1), y)
        return model

    def _isotonic(self, p: np.ndarray, y: np.ndarray):
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return model

    def _histogram_binning(self, p: np.ndarray, y: np.ndarray, bins: int = 15):
        """
        Histogram binning calibration.
        """
        edges = np.linspace(0, 1, bins + 1)
        bin_idx = np.digitize(p, edges) - 1
        bin_means = {}

        for b in range(bins):
            mask = bin_idx == b
            if np.sum(mask) > 0:
                bin_means[b] = float(np.mean(y[mask]))
            else:
                bin_means[b] = 0.5

        return HistogramBinningCalibrator(edges, bin_means)

    def _apply_calibrator(self, cal, p: np.ndarray) -> np.ndarray:
        """
        Applies platt/isotonic/binning to probability array.
        """

        if hasattr(cal, "predict_proba"):
            # logistic regression (Platt)
            out = cal.predict_proba(p.reshape(-1, 1))[:, 1]
            return out

        if hasattr(cal, "predict"):
            # isotonic regression
            return cal.predict(p)

        # histogram binning
        return cal.predict(p)

    def calibrate(self, p_raw: np.ndarray, y_true: np.ndarray) -> Any:
        """
        Fit calibrator according to method or choose best automatically.
        Returns the chosen calibrator object.
        """

        if self.method != "auto":
            return self._manual(p_raw, y_true)

        # auto-selection
        cands = {}

        # Platt
        pl = self._platt(p_raw, y_true)
        cands["platt"] = pl

        # Isotonic
        iso = self._isotonic(p_raw, y_true)
        cands["isotonic"] = iso

        # Histogram
        hist = self._histogram_binning(p_raw, y_true)
        cands["histogram"] = hist

        # Score all
        best_score = 1e9
        best_key = None
        best_cal = None

        for k, cal in cands.items():
            pred = self._apply_calibrator(cal, p_raw)
            score = brier_score_loss(y_true, pred)
            if score < best_score:
                best_score = score
                best_key = k
                best_cal = cal

        log(f"Best calibration method: {best_key}, Brier={best_score:.6f}")
        return best_cal

    def _manual(self, p_raw: np.ndarray, y_true: np.ndarray) -> Any:
        """
        Manual calibration mode.
        """
        if self.method == "platt":
            return self._platt(p_raw, y_true)
        if self.method == "isotonic":
            return self._isotonic(p_raw, y_true)
        if self.method == "histogram":
            return self._histogram_binning(p_raw, y_true)
        raise ValueError(f"Unknown calibration method: {self.method}")
