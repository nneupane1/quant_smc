"""
CoolingEngine:
    Implements Adaptive Cooling + Moonshot Override + 2R+ Override.

    Cooling period:
        - Activated after locking significant profit.
        - Duration = min_bars + fraction_locked * max_bars
        - Fraction_locked = lock_amount / (current_equity - base_equity)

    During cooling period:
        - Confluence thresholds increase
        - EVR thresholds increase
        - Hazard cap tightens
        - EMA stretch guard tightens
        - System trades ONLY if:
              A) moonshot override triggers
              B) 2R+ override triggers (less strict)

    Never uses fixed numbers inside code.
"""

from typing import Dict, Any
from quant_system.utils.logger import log


def _as_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class CoolingEngine:
    """
    Central logic for cooling management.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = _as_dict(config)
        ccfg = cfg.get("execution", {}).get("cooling", {})

        self.min_bars = int(ccfg.get("min_bars", 4))          # e.g., 4
        self.max_bars = int(ccfg.get("max_bars", 20))         # e.g., 20

        # Threshold increases during cooling
        self.conf_boost = float(ccfg.get("conf_boost", 0.15))    # e.g., +0.15
        self.evr_boost  = float(ccfg.get("evr_boost", 0.30))     # e.g., +0.30
        self.hazard_cut = float(ccfg.get("hazard_cut", -0.10))   # e.g., -0.10

        # Override parameters
        self.moon_conf_add = float(ccfg.get("moon_conf_add", 0.10))
        self.moon_evr_min = float(ccfg.get("moon_evr_min", 1.80))
        self.moon_medR_min = float(ccfg.get("moon_medR_min", 6.0))
        self.moon_hazard_max = float(ccfg.get("moon_hazard_max", 0.25))

        self.twoR_conf_add = float(ccfg.get("twoR_conf_add", 0.05))
        self.twoR_evr_min = float(ccfg.get("twoR_evr_min", 1.20))
        self.twoR_medR_min = float(ccfg.get("twoR_medR_min", 2.0))
        self.twoR_hazard_max = float(ccfg.get("twoR_hazard_max", 0.40))

        log("CoolingEngine initialized.")

    # ------------------------------------------------------------------
    def compute_cooldown(self, current_equity: float, base_equity: float, locked_amount: float) -> int:
        """
        Adaptive cooldown duration in bars.
        """
        excess = max(current_equity - base_equity, 1e-9)

        locked_fraction = locked_amount / excess
        locked_fraction = max(0.0, min(1.0, locked_fraction))

        duration = self.min_bars + int(self.max_bars * locked_fraction)
        duration = max(self.min_bars, min(self.min_bars + self.max_bars, duration))

        log(
            f"CoolingEngine: computed cooldown={duration}, "
            f"locked_fraction={locked_fraction:.3f}, locked_amount={locked_amount:.2f}"
        )
        return duration

    # ------------------------------------------------------------------
    def adjust_thresholds(self, conf_thr: float, evr_thr: float, hazard_cap: float) -> Dict[str, float]:
        """
        Apply cooling penalties to thresholds.
        """
        return {
            "conf": conf_thr + self.conf_boost,
            "evr":  evr_thr + self.evr_boost,
            "hazard": max(0.0, hazard_cap + self.hazard_cut),
        }

    # ------------------------------------------------------------------
    def allow_moonshot_override(self, conf: float, base_conf_thr: float,
                                evr: float, medianR: float, hazard: float) -> bool:
        """
        Moonshot override: Allows trade during cooldown if extremely high quality.
        """
        if conf < base_conf_thr + self.moon_conf_add:
            return False
        if evr < self.moon_evr_min:
            return False
        if medianR < self.moon_medR_min:
            return False
        if hazard > self.moon_hazard_max:
            return False

        log("CoolingEngine: MOONSHOT override triggered.")
        return True

    # ------------------------------------------------------------------
    def allow_twoR_override(self, conf: float, base_conf_thr: float,
                            evr: float, medianR: float, hazard: float) -> bool:
        """
        2R+ override: Allows trade during cooldown if reasonably strong.
        """
        if conf < base_conf_thr + self.twoR_conf_add:
            return False
        if evr < self.twoR_evr_min:
            return False
        if medianR < self.twoR_medR_min:
            return False
        if hazard > self.twoR_hazard_max:
            return False

        log("CoolingEngine: 2R+ override triggered.")
        return True

    # ------------------------------------------------------------------
    def cooling_gate(
        self,
        in_cooldown: bool,
        cooldown_remaining: int,
        conf: float,
        base_conf_thr: float,
        evr: float,
        medianR: float,
        hazard: float
    ) -> Dict[str, Any]:
        """
        Decide whether a trade is permitted during cooling.

        Returns:
            {
                "allow": bool,
                "override": "moonshot" / "2R" / None
            }
        """
        if not in_cooldown:
            return {"allow": True, "override": None}

        # Moonshot override (first priority)
        if self.allow_moonshot_override(conf, base_conf_thr, evr, medianR, hazard):
            return {"allow": True, "override": "moonshot"}

        # 2R override (second priority)
        if self.allow_twoR_override(conf, base_conf_thr, evr, medianR, hazard):
            return {"allow": True, "override": "2R"}

        # Otherwise: no trade permitted
        log(f"CoolingEngine: trade blocked, cooldown_remaining={cooldown_remaining}")
        return {"allow": False, "override": None}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "min_bars": self.min_bars,
            "max_bars": self.max_bars,
            "conf_boost": self.conf_boost,
            "evr_boost": self.evr_boost,
            "hazard_cut": self.hazard_cut,
        }
