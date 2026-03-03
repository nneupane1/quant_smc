import pandas as pd
from typing import Any, Dict
from quant_system.utils.logger import log


def _as_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(config, "load"):
        return config.load()
    if hasattr(config, "full"):
        return config.full
    return dict(config)


class ExposureTracker:
    """
    Tracks exposure and enforces risk constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = _as_dict(config)
        ecfg = cfg.get("execution", {}).get("exposure_tracker", {})

        self.borrow_limit = float(ecfg.get("borrow_limit", 0.0))
        self.max_gross_mult = float(ecfg.get("max_gross_exposure_mult", 1.0))

        self.by_asset: Dict[str, Dict[str, float]] = {}
        self.state = self._aggregate_state()

        self.history = []  # time series for dashboards

        log("ExposureTracker initialized.")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _asset_bucket(self, asset: str) -> Dict[str, float]:
        asset = asset or "GLOBAL"
        if asset not in self.by_asset:
            self.by_asset[asset] = {"spot_long": 0.0, "dir_short": 0.0, "hedge_short": 0.0}
        return self.by_asset[asset]

    def _aggregate_state(self) -> Dict[str, float]:
        spot_long = sum(v["spot_long"] for v in self.by_asset.values())
        dir_short = sum(v["dir_short"] for v in self.by_asset.values())
        hedge_short = sum(v["hedge_short"] for v in self.by_asset.values())
        return {
            "spot_long": spot_long,
            "dir_short": dir_short,
            "hedge_short": hedge_short,
        }

    def _gross(self) -> float:
        self.state = self._aggregate_state()
        long = abs(self.state["spot_long"])
        short = abs(self.state["dir_short"] + self.state["hedge_short"])
        return long + short

    def _net(self) -> float:
        self.state = self._aggregate_state()
        return self.state["spot_long"] - (self.state["dir_short"] + self.state["hedge_short"])

    def _leverage(self, equity: float) -> float:
        g = self._gross()
        return g / equity if equity > 0 else 0.0

    # ------------------------------------------------------------
    # Update exposures
    # ------------------------------------------------------------
    def register_long(self, notional: float, asset: str = "GLOBAL"):
        bucket = self._asset_bucket(asset)
        bucket["spot_long"] += notional
        self.state = self._aggregate_state()
        log(f"Exposure: Added long {notional:.2f} on {asset}, total_long={self.state['spot_long']:.2f}")

    def register_short(self, notional: float, asset: str = "GLOBAL"):
        """
        Directional short.
        Limited by borrow_limit.
        """
        bucket = self._asset_bucket(asset)
        new_short = self.state["dir_short"] + notional
        if new_short > self.borrow_limit:
            notional = max(0.0, self.borrow_limit - self.state["dir_short"])
            log(f"Exposure: Borrow limit reached, restricting directional short to {notional:.2f}")

        bucket["dir_short"] += notional
        self.state = self._aggregate_state()
        log(f"Exposure: Added directional short {notional:.2f}, total_dir_short={self.state['dir_short']:.2f}")

    def register_hedge(self, notional: float, parent_notional: float, asset: str = "GLOBAL"):
        """
        Hedge sizing: cannot exceed parent delta.
        """
        allowed = max(0.0, parent_notional)
        alloc = min(notional, allowed)

        bucket = self._asset_bucket(asset)
        bucket["hedge_short"] += alloc
        self.state = self._aggregate_state()
        log(f"Exposure: Added hedge short {alloc:.2f}, total_hedge={self.state['hedge_short']:.2f}")

    def release(self, asset: str, *, long_notional: float = 0.0, short_notional: float = 0.0, hedge_notional: float = 0.0):
        bucket = self._asset_bucket(asset)
        bucket["spot_long"] = max(0.0, bucket["spot_long"] - long_notional)
        bucket["dir_short"] = max(0.0, bucket["dir_short"] - short_notional)
        bucket["hedge_short"] = max(0.0, bucket["hedge_short"] - hedge_notional)
        self.state = self._aggregate_state()

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

    def current_exposures(self, equity: float = 0.0) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for asset, bucket in self.by_asset.items():
            long_val = float(bucket["spot_long"])
            short_val = float(bucket["dir_short"] + bucket["hedge_short"])
            gross = abs(long_val) + abs(short_val)
            out[asset] = {
                "long": long_val,
                "short": short_val,
                "net": long_val - short_val,
                "gross": gross,
                "risk_weight": (gross / equity) if equity > 0 else gross,
            }
        return out

    # ------------------------------------------------------------
    # Record history
    # ------------------------------------------------------------
    def snapshot(self, timestamp, equity: float):
        self.state = self._aggregate_state()
        entry = {
            "timestamp": timestamp,
            "spot_long": self.state["spot_long"],
            "dir_short": self.state["dir_short"],
            "hedge_short": self.state["hedge_short"],
            "gross": self._gross(),
            "net": self._net(),
            "leverage": self._leverage(equity),
            "equity": equity,
            "by_asset": self.current_exposures(equity),
        }
        self.history.append(entry)

    # ------------------------------------------------------------
    # Utility accessors for dashboards
    # ------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
