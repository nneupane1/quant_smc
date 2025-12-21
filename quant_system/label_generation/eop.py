"""
EOP Label Generator
-------------------

Expected Opportunity Probability label.

Definition:

    Label = 1 if AN A+ opportunity appears within the next
             H = 96 × 15m bars (24 hours), based entirely on
             fully closed bars and precomputed quality metrics.

    A+ conditions (from system spec):
        - conf >= tau_A_plus
        - evr >= 1.5
        - median_r >= 5.0
        - hazard <= 0.35
        - higher-TF gates allow the direction
"""

from typing import Dict, List, Optional
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log
from quant_system.config.config_loader import ConfigLoader


class EOPLabeler:
    """
    Generates EOP labels using confluence/EVR/hazard precomputed metrics.

    Parameters:
        horizon_bars: default = 96 (24 hours)
        tau_A_plus: A+ tier confidence threshold (from calibration)
        evr_min: minimum EVR for A+
        median_r_min: minimum median-R requirement
        hazard_max: maximum hazard allowed
    """

    def __init__(
        self,
        horizon_bars: int = 96,
        tau_A_plus: float = 0.70,
        evr_min: float = 1.5,
        median_r_min: float = 5.0,
        hazard_max: float = 0.35
    ):
        self.horizon_bars = horizon_bars
        self.tau_A_plus = tau_A_plus
        self.evr_min = evr_min
        self.median_r_min = median_r_min
        self.hazard_max = hazard_max

        log(
            "EOPLabeler initialized "
            f"(horizon_bars={horizon_bars}, "
            f"tau_A_plus={tau_A_plus}, evr_min={evr_min}, "
            f"median_r_min={median_r_min}, hazard_max={hazard_max})."
        )

    def generate_labels(
        self,
        candles: List[Candle],
        conf_scores: Dict[int, float],
        evr_scores: Dict[int, float],
        median_r: Dict[int, float],
        hazard: Dict[int, float],
        tf_gates: Dict[int, Dict[str, float]]
    ) -> Dict[int, int]:
        """
        Generate EOP labels.

        Inputs:
            conf_scores[ts]: learned confluence
            evr_scores[ts]: expected R
            median_r[ts]: median R
            hazard[ts]: hazard estimate
            tf_gates[ts]: dict with { "allow_long": bool, "allow_short": bool }

        Output:
            ts → 0/1
        """

        log("Generating EOP labels.")

        ts_arr = [c.timestamp for c in candles]
        idx = {ts: i for i, ts in enumerate(ts_arr)}
        N = len(ts_arr)

        labels: Dict[int, int] = {}

        for i, ts in enumerate(ts_arr):

            end = min(N, i + self.horizon_bars + 1)
            success = 0

            # Scan forward for any A+ opportunity
            for j in range(i + 1, end):
                ts_f = ts_arr[j]

                conf = conf_scores.get(ts_f, None)
                evr = evr_scores.get(ts_f, None)
                medr = median_r.get(ts_f, None)
                haz = hazard.get(ts_f, None)
                gates = tf_gates.get(ts_f, {})

                if conf is None or evr is None or medr is None or haz is None:
                    continue

                if conf < self.tau_A_plus:
                    continue
                if evr < self.evr_min:
                    continue
                if medr < self.median_r_min:
                    continue
                if haz > self.hazard_max:
                    continue

                # Check structural TF gates (10h/6h/1h)
                allow_l = gates.get("allow_long", False)
                allow_s = gates.get("allow_short", False)

                # If either long or short A+ matches, mark success
                if allow_l or allow_s:
                    success = 1
                    break

            labels[ts] = success

        log(f"EOP label generation complete. Labels: {len(labels)}")
        return labels

    # ------------------------------------------------------------------
    def apply(self, df15: pd.DataFrame, cfg_loader: Optional[ConfigLoader] = None) -> pd.DataFrame:
        """
        Attach label_eop to a 15m dataframe based on A+ occurrence within horizon.
        Expects columns: conf_score/evr/median_r/hazard or similar.
        """
        H = self.horizon_bars
        tau = self.tau_A_plus
        evr_min = self.evr_min
        medr_min = self.median_r_min
        haz_max = self.hazard_max
        if cfg_loader:
            lc = cfg_loader.load_yaml("labels.yaml")["labels"]["eop"]
            H = int(lc.get("horizon_bars", H))
            tau = float(lc.get("Aplus_min_evr", evr_min))
            evr_min = float(lc.get("Aplus_min_evr", evr_min))
            medr_min = float(lc.get("Aplus_min_medianR", medr_min))
            haz_max = float(lc.get("hazard_cap", haz_max))

        df = df15.copy()
        labels = []
        for i, row in df.iterrows():
            window = df.iloc[i + 1:i + 1 + H]
            ok = False
            if not window.empty:
                conf_series = window.get("conf_score", pd.Series(0))
                evr_series = window.get("evr", pd.Series(0))
                medr_series = window.get("median_r", pd.Series(0))
                haz_series = window.get("hazard", pd.Series(0))
                ok = ((conf_series >= tau) & (evr_series >= evr_min) & (medr_series >= medr_min) & (haz_series <= haz_max)).any()
            labels.append(int(ok))
        df["label_eop"] = labels
        return df
