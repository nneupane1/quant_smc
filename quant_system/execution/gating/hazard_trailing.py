"""
HazardTrailingEngine:
    Survival-model-driven trailing logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from quant_system.utils.logger import get_logger

LOG = get_logger("hazard_trailing")


def _as_dict(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class HazardTrailingEngine:
    """
    Implements hazard-based optimal stopping.
    """

    def __init__(self, config: Union[Dict[str, Any], Any]):
        cfg = _as_dict(config)
        hcfg = cfg.get("execution", {}).get("hazard_trailing", {})

        self.thresholds = hcfg.get("thresholds", {})
        self.runner_thresholds = hcfg.get("runner_thresholds", self.thresholds)

        atr_mult_cfg = hcfg.get("atr_multipliers", {"tighten": 0.5, "exit_band": 1.0})
        self.atr_mult_tighten = float(atr_mult_cfg.get("tighten", 0.5))
        self.atr_mult_exitband = float(atr_mult_cfg.get("exit_band", 1.0))

        runner_mult_cfg = hcfg.get("runner_multipliers", {})
        self.runner_atr_mult_tighten = float(runner_mult_cfg.get("tighten", self.atr_mult_tighten))
        self.runner_atr_mult_exitband = float(runner_mult_cfg.get("exit_band", self.atr_mult_exitband))

        self.ema_stretch_z = float(hcfg.get("ema_stretch_z", 3.0))
        self.decay_bars = int(hcfg.get("freshness_decay_bars", 32))
        self.seg_thresholds = hcfg.get("segment_thresholds", {})
        self.runner_moonshot = hcfg.get("runner_moonshot", {})

    def _segment_override(self, row: pd.Series, hazard: float) -> float:
        regime = row.get("regime_class", "default")
        session = row.get("session", "other")
        seg_cfg = self.seg_thresholds.get(regime, {}).get(session, {})
        if "exit" in seg_cfg and hazard > seg_cfg["exit"]:
            return float(seg_cfg["exit"]) * 1.01
        return hazard

    def _adjust_for_staleness(self, hazard: float, bars_in_trade: int) -> float:
        if bars_in_trade > self.decay_bars:
            return hazard + 0.02 * ((bars_in_trade - self.decay_bars) / self.decay_bars)
        return hazard

    def _ema_stretch_adjust(self, hazard: float, row: pd.Series) -> float:
        z = float(row.get("ema_dist_z", 0.0))
        if abs(z) > self.ema_stretch_z:
            return hazard + 0.05 * (abs(z) - self.ema_stretch_z)
        return hazard

    def _runner_moonshot_adjust(
        self,
        row: pd.Series,
        hazard: float,
        tighten_thr: float,
        partial_thr: float,
        exit_thr: float,
        leg: str,
    ) -> Dict[str, float]:
        if leg != "runner" or not self.runner_moonshot.get("enabled", False):
            return {
                "hazard": hazard,
                "tighten_thr": tighten_thr,
                "partial_thr": partial_thr,
                "exit_thr": exit_thr,
            }

        conf = float(row.get("confluence_score", row.get("conf_score", 0.0)) or 0.0)
        bos_cont = float(row.get("p_bos_cont", row.get("prob_bos_cont", 0.0)) or 0.0)
        eop = float(row.get("p_eop", row.get("prob_eop", 0.0)) or 0.0)
        edp = float(row.get("p_edp", row.get("prob_edp", 0.0)) or 0.0)
        median_r = float(row.get("median_r", row.get("median_R", 0.0)) or 0.0)

        if conf < float(self.runner_moonshot.get("confluence_min", 0.75)):
            return {
                "hazard": hazard,
                "tighten_thr": tighten_thr,
                "partial_thr": partial_thr,
                "exit_thr": exit_thr,
            }
        if bos_cont < float(self.runner_moonshot.get("bos_cont_min", 0.60)):
            return {
                "hazard": hazard,
                "tighten_thr": tighten_thr,
                "partial_thr": partial_thr,
                "exit_thr": exit_thr,
            }
        if (eop - edp) < float(self.runner_moonshot.get("eop_edge_min", 0.0)):
            return {
                "hazard": hazard,
                "tighten_thr": tighten_thr,
                "partial_thr": partial_thr,
                "exit_thr": exit_thr,
            }
        if median_r < float(self.runner_moonshot.get("median_r_min", 6.0)):
            return {
                "hazard": hazard,
                "tighten_thr": tighten_thr,
                "partial_thr": partial_thr,
                "exit_thr": exit_thr,
            }

        return {
            "hazard": max(0.0, hazard - float(self.runner_moonshot.get("hazard_discount", 0.08))),
            "tighten_thr": tighten_thr + float(self.runner_moonshot.get("tighten_bonus", 0.03)),
            "partial_thr": partial_thr + float(self.runner_moonshot.get("partial_bonus", 0.05)),
            "exit_thr": exit_thr + float(self.runner_moonshot.get("exit_bonus", 0.08)),
        }

    def _tighten_stop(self, side: str, px: float, atr: float, mult: float) -> float:
        if atr <= 0:
            return px
        return px - mult * atr if side == "long" else px + mult * atr

    def _exit_band(self, side: str, px: float, atr: float, mult: float) -> float:
        if atr <= 0:
            return px
        return px - mult * atr if side == "long" else px + mult * atr

    def evaluate(
        self,
        row: pd.Series,
        hazard: float,
        side: str,
        bars_in_trade: int,
        current_stop: float,
        config: Dict[str, Any] = None,
        leg: str = "core",
    ) -> Dict[str, Any]:
        """
        Returns dict:
            action: one of {hold, tighten, partial, exit}
            new_stop: updated stop
        """
        px = float(row["close"])
        atr = float(row.get("atr", np.nan))

        thr = self.runner_thresholds if leg == "runner" else self.thresholds
        tighten_thr = float(thr.get("tighten", 0.3))
        partial_thr = float(thr.get("partial", 0.4))
        exit_thr = float(thr.get("exit", 0.5))
        hold_thr = float(thr.get("hold", 0.2))

        atr_tighten = self.runner_atr_mult_tighten if leg == "runner" else self.atr_mult_tighten
        atr_exitband = self.runner_atr_mult_exitband if leg == "runner" else self.atr_mult_exitband

        hazard_adj = self._segment_override(row, hazard)
        hazard_adj = self._adjust_for_staleness(hazard_adj, bars_in_trade)
        hazard_adj = self._ema_stretch_adjust(hazard_adj, row)
        runner_adj = self._runner_moonshot_adjust(
            row=row,
            hazard=hazard_adj,
            tighten_thr=tighten_thr,
            partial_thr=partial_thr,
            exit_thr=exit_thr,
            leg=leg,
        )
        hazard_adj = runner_adj["hazard"]
        tighten_thr = runner_adj["tighten_thr"]
        partial_thr = runner_adj["partial_thr"]
        exit_thr = runner_adj["exit_thr"]

        if hazard_adj > exit_thr:
            new_stop = self._exit_band(side, px, atr, atr_exitband)
            return {"action": "exit", "new_stop": new_stop}

        if hazard_adj > partial_thr:
            new_stop = self._tighten_stop(side, px, atr, atr_tighten)
            return {"action": "partial", "new_stop": new_stop}

        if hazard_adj > tighten_thr:
            new_stop = self._tighten_stop(side, px, atr, atr_tighten)
            return {"action": "tighten", "new_stop": new_stop}

        if hazard_adj <= hold_thr:
            return {"action": "hold", "new_stop": current_stop}

        return {"action": "hold", "new_stop": current_stop}
