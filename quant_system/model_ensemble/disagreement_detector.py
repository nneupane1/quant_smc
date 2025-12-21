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

        rows = []
        for name, out in raw_outputs.items():
            if out:
                rows.append(list(out.values()))

        if not rows:
            return {"spread": 0, "max_gap": 0, "pairwise": 0}

        arr = np.array(rows)

        # pairwise variance across models
        spread = float(arr.var())

        # max difference on any specialist
        max_gap = float((arr.max(axis=0) - arr.min(axis=0)).max())

        return {
            "spread": spread,
            "max_gap": max_gap,
            "pairwise": float(np.mean(np.abs(arr[:, None, :] - arr[None, :, :])))
        }
