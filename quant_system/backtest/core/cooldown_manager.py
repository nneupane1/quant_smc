"""
Adaptive Cooling Manager
Manages post-trade cooldown periods, moonshot overrides,
hazard-weighted acceleration, and regime-based relaxation.
Used by both backtester and live orchestrator.

Rules implemented:
  - Base cooling = config-driven (e.g., 2×15m bars)
  - Adaptive extension after loss streak
  - Hazard-weighted contraction after clean strong winners
  - 2R Early-Continuation override: allow immediate re-entry if setup is ≥2R and hazard low
  - Moonshot unlock: ≥5R winner resets cooldown to zero
  - Regime accelerator: trend regime reduces cooldown; chop regime increases it
  - Trade-rate smoothing: slow down if >5 trades/day rolling
"""

from dataclasses import dataclass
from quant_system.utils.logger import get_logger
import numpy as np
import datetime as dt

LOG = get_logger("cooldown_manager")


@dataclass
class CoolingState:
    last_trade_time: dt.datetime = None
    cooldown_until: dt.datetime = None
    loss_streak: int = 0
    rolling_trades_today: int = 0
    rolling_wins_today: int = 0
    rolling_losses_today: int = 0


class CooldownManager:
    def __init__(
        self,
        base_cooldown_minutes: int,
        max_daily_trades: int,
        hazard_relax_threshold: float,
        hazard_extend_threshold: float,
        moonshot_r: float = 5.0,
        continuation_r: float = 2.0,
    ):
        """
        base_cooldown_minutes: baseline cooling interval after a trade
        max_daily_trades: soft cap for daily trading frequency
        hazard_relax_threshold: hazard < X shrinks cooldown
        hazard_extend_threshold: hazard > Y extends cooldown
        moonshot_r: ≥ this R resets cooldown completely
        continuation_r: if winner ≥ continuation_r & hazard low → no cooldown
        """
        self.base = base_cooldown_minutes
        self.max_daily = max_daily_trades
        self.hazard_relax = hazard_relax_threshold
        self.hazard_extend = hazard_extend_threshold
        self.moonshot_r = moonshot_r
        self.cont_r = continuation_r

    # ---------------------------------------------------------
    # Decision Interface
    # ---------------------------------------------------------
    def is_cooled(self, now: dt.datetime, state: CoolingState) -> bool:
        if state.cooldown_until is None:
            return True
        return now >= state.cooldown_until

    # ---------------------------------------------------------
    # Main Update Entry
    # ---------------------------------------------------------
    def update_after_trade(
        self,
        now: dt.datetime,
        state: CoolingState,
        realized_r: float,
        hazard_at_exit: float,
        regime: str,
    ):
        """
        Update cooldown after a trade.
        Applies moonshot rule, continuation rule, hazard rules, streak rules, and regime modifiers.
        """
        LOG.info(f"[Cooldown Manager] Updating cooldown after trade")
        LOG.info(f"  realized R: {realized_r:.2f}")
        LOG.info(f"  hazard_at_exit: {hazard_at_exit:.3f}")
        LOG.info(f"  regime: {regime}")
        LOG.info(f"  current loss streak: {state.loss_streak}")

        # Track rolling performance
        if realized_r >= 0:
            state.rolling_wins_today += 1
            state.loss_streak = 0
        else:
            state.rolling_losses_today += 1
            state.loss_streak += 1

        state.rolling_trades_today += 1
        state.last_trade_time = now

        # -----------------------------------------------------
        # Moonshot Reset (≥5R) → zero cooldown
        # -----------------------------------------------------
        if realized_r >= self.moonshot_r:
            LOG.info("  Moonshot achieved → cooldown reset to zero")
            state.cooldown_until = now
            return

        # -----------------------------------------------------
        # 2R Early Continuation Override (low hazard)
        # -----------------------------------------------------
        if realized_r >= self.cont_r and hazard_at_exit < self.hazard_relax:
            LOG.info("  2R continuation override triggered → no cooldown")
            state.cooldown_until = now
            return

        # -----------------------------------------------------
        # Compute baseline cooldown
        # -----------------------------------------------------
        cooldown_minutes = self.base
        LOG.info(f"  base cooldown: {cooldown_minutes} min")

        # -----------------------------------------------------
        # Hazard-weighted adjustments
        # -----------------------------------------------------
        if hazard_at_exit < self.hazard_relax:
            cooldown_minutes *= 0.6
            LOG.info(f"  hazard low → cooldown reduced to {cooldown_minutes}")
        elif hazard_at_exit > self.hazard_extend:
            cooldown_minutes *= 1.5
            LOG.info(f"  hazard high → cooldown extended to {cooldown_minutes}")

        # -----------------------------------------------------
        # Loss streak adjustment
        # -----------------------------------------------------
        if state.loss_streak >= 2:
            cooldown_minutes *= 1.3 + 0.2 * (state.loss_streak - 1)
            LOG.info(f"  loss streak {state.loss_streak} → cooldown raised to {cooldown_minutes}")

        # -----------------------------------------------------
        # Regime modifiers
        # -----------------------------------------------------
        if regime == "trend":
            cooldown_minutes *= 0.8
            LOG.info("  trend regime → cooldown reduced")
        elif regime in ("chop", "range"):
            cooldown_minutes *= 1.25
            LOG.info("  chop/range regime → cooldown increased")

        # -----------------------------------------------------
        # Daily trade-rate smoothing
        # -----------------------------------------------------
        if state.rolling_trades_today > self.max_daily:
            factor = 1.0 + (state.rolling_trades_today - self.max_daily) * 0.20
            cooldown_minutes *= factor
            LOG.info(f"  exceeded daily trade target → cooldown increased to {cooldown_minutes}")

        # Clamp
        cooldown_minutes = max(5, min(cooldown_minutes, 240))
        LOG.info(f"  final cooldown duration: {cooldown_minutes} min")

        state.cooldown_until = now + dt.timedelta(minutes=cooldown_minutes)

    # ---------------------------------------------------------
    # Hard reset for new trading day
    # ---------------------------------------------------------
    def reset_daily(self, state: CoolingState):
        LOG.info("Resetting daily cooldown counters")
        state.rolling_trades_today = 0
        state.rolling_wins_today = 0
        state.rolling_losses_today = 0
        state.loss_streak = 0
