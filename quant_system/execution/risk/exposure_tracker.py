"""
ExposureTracker:
    Maintains real-time portfolio exposure:
        - spot long exposure
        - directional short exposure (spot-margin)
        - hedge short exposure (perp)
        - gross exposure
        - net exposure
        - leverage
        - borrow usage for shorting
        - exposure-time series for dashboards

Constraints enforced:
    - |long| + |short| <= 2 * equity
    - directional short <= borrow_limit
    - hedge short <= parent_position_delta

Used by:
    - backtester
    - forward engine
    - live system
    - dashboards
"""

import pandas as pd
from typing import Dict, Any
from quant_system.utils.logger import log


class ExposureTracker:
    """
    Tracks exposure and enforces risk constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        ecfg = config["execution"]["exposure_tracker"]

        self.borrow_limit = float(ecfg["borrow_limit"])
        self.max_gross_mult = float(ecfg["max_gross_exposure_mult"])

        # Current exposure state
        self.state = {
            "spot_long": 0.0,       # notional
            "dir_short": 0.0,       # directional short notional
            "hedge_short": 0.0,     # hedge short notional (perp)
        }

        self.history = []  # time series for dashboards

        log("ExposureTracker initialized.")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _gross(self) -> float:
        long = abs(self.state["spot_long"])
        short = abs(self.state["dir_short"] + self.state["hedge_short"])
        return long + short

    def _net(self) -> float:
        return self.state["spot_long"] - (self.state["dir_short"] + self.state["hedge_short"])

    def _leverage(self, equity: float) -> float:
        g = self._gross()
        return g / equity if equity > 0 else 0.0

    # ------------------------------------------------------------
    # Update exposures
    # ------------------------------------------------------------
    def register_long(self, notional: float):
        self.state["spot_long"] += notional
        log(f"Exposure: Added long {notional:.2f}, total_long={self.state['spot_long']:.2f}")

    def register_short(self, notional: float):
        """
        Directional short.
        Limited by borrow_limit.
        """
        new_short = self.state["dir_short"] + notional
        if new_short > self.borrow_limit:
            notional = max(0.0, self.borrow_limit - self.state["dir_short"])
            log(f"Exposure: Borrow limit reached, restricting directional short to {notional:.2f}")

        self.state["dir_short"] += notional
        log(f"Exposure: Added directional short {notional:.2f}, total_dir_short={self.state['dir_short']:.2f}")

    def register_hedge(self, notional: float, parent_notional: float):
        """
        Hedge sizing: cannot exceed parent delta.
        """
        allowed = max(0.0, parent_notional)
        alloc = min(notional, allowed)

        self.state["hedge_short"] += alloc
        log(f"Exposure: Added hedge short {alloc:.2f}, total_hedge={self.state['hedge_short']:.2f}")

    # ------------------------------------------------------------
    # Exposure feasibility check
    # ------------------------------------------------------------
    def enforce_limits(self, equity: float) -> bool:
        """
        Returns True if within limits, False if breaches occurred.
        """
        gross = self._gross()
        if gross > self.max_gross_mult * equity:
            log(f"Exposure breach: gross={gross:.2f} > {self.max_gross_mult} * equity")
            return False

        return True

    # ------------------------------------------------------------
    # Record history
    # ------------------------------------------------------------
    def snapshot(self, timestamp, equity: float):
        entry = {
            "timestamp": timestamp,
            "spot_long": self.state["spot_long"],
            "dir_short": self.state["dir_short"],
            "hedge_short": self.state["hedge_short"],
            "gross": self._gross(),
            "net": self._net(),
            "leverage": self._leverage(equity),
            "equity": equity,
        }
        self.history.append(entry)

    # ------------------------------------------------------------
    # Utility accessors for dashboards
    # ------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
