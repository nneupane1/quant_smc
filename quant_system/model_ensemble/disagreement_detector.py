"""
disagreement_detector.py
Quantifies how much models disagree.
Used for:
 • Hazard tightening
 • MPC hedge increase
 • Cooling mode activation
"""

import numpy as np


class DisagreementDetector:

    def evaluate(self, raw_outputs):
        """
        raw_outputs = { model_name : { specialist: prob } }
        """

        keys = sorted(
            {
                key
                for out in raw_outputs.values()
                if isinstance(out, dict)
                for key, value in out.items()
                if isinstance(value, (int, float, np.number))
            }
        )
        if not keys:
            return {"spread": 0.0, "max_gap": 0.0, "pairwise": 0.0, "keys": []}

        rows = []
        for out in raw_outputs.values():
            if not isinstance(out, dict):
                continue
            rows.append([float(out.get(key, np.nan)) for key in keys])

        if not rows:
            return {"spread": 0.0, "max_gap": 0.0, "pairwise": 0.0, "keys": keys}

        arr = np.array(rows, dtype=float)
        with np.errstate(invalid="ignore"):
            spread = float(np.nanvar(arr))
            gaps = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0) if arr.size else np.array([0.0])
            max_gap = float(np.nanmax(gaps)) if gaps.size else 0.0
            pairwise = float(np.nanmean(np.abs(arr[:, None, :] - arr[None, :, :]))) if arr.size else 0.0

        return {
            "spread": spread,
            "max_gap": max_gap,
            "pairwise": pairwise,
            "keys": keys,
        }
