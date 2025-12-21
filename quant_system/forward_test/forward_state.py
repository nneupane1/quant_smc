"""
forward_state.py

Maintains all live paper-trading state for Forward Testing:
 - virtual equity (compounding)
 - locked equity bucket (via MPC)
 - available_risk_capital (dynamic)
 - open trades
 - hedges
 - cooling periods
 - moonshot override logic
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from quant_system.utils.logger import get_logger

LOG = get_logger("forward_state")


@dataclass
class TradeState:
    """Represents an open trade in forward test."""
    trade_id: str
    side: str  # "long" or "short"
    entry_price: float
    stop_price: float
    size: float  # position size in quote currency
    timestamp: datetime
    risk_perc: float
    r_mult: float = 0.0
    highest_r: float = 0.0
    is_hedge: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardState:
    """Holds dynamic capital + open positions during forward test."""

    # capital model
    starting_capital: float = 20_000.0
    equity: float = 20_000.0
    locked_profit: float = 0.0
    free_capital: float = 20_000.0

    # strategy state
    open_trades: Dict[str, TradeState] = field(default_factory=dict)
    cooling_end_time: Optional[datetime] = None

    # MPC dynamic risk parameters
    current_risk_mode: float = 0.01  # 1% default
    current_hedge_ratio: float = 0.0

    # moonshot override logic
    moonshot_min_r: float = 2.0

    # bookkeeping
    last_update_ts: Optional[datetime] = None

    # ---------------------------------------------------------------
    def log_state(self):
        LOG.info(f"ForwardState | equity={self.equity:.2f} "
                 f"locked={self.locked_profit:.2f} free={self.free_capital:.2f} "
                 f"risk={self.current_risk_mode:.3f} hedge={self.current_hedge_ratio:.2f} "
                 f"open_trades={len(self.open_trades)}")

    # ---------------------------------------------------------------
    def start_cooling(self, until_ts: datetime):
        """Begin a cooling period where entries are suppressed unless moonshot override fires."""
        self.cooling_end_time = until_ts
        LOG.info(f"Cooling period started → until {until_ts}.")

    # ---------------------------------------------------------------
    def cooling_active(self, now: datetime) -> bool:
        if self.cooling_end_time is None:
            return False
        return now < self.cooling_end_time

    # ---------------------------------------------------------------
    def can_enter_trade(self, conf: float, evr: float, median_r: float, hazard: float,
                        moonshot_flag: bool, now: datetime) -> bool:
        """Entry gate after all execution logic; includes cooling + moonshot override."""

        if not self.cooling_active(now):
            return True  # free to trade normally

        # cooling active → only moonshots allowed
        if moonshot_flag and median_r >= self.moonshot_min_r:
            LOG.info("Moonshot override triggered during cooling → trade allowed.")
            return True

        LOG.info("Cooling active → normal entries suppressed.")
        return False

    # ---------------------------------------------------------------
    def update_for_mpc(self, lock_fraction: float, risk_mode: float, hedge_ratio: float):
        """
        Apply MPC outputs:
         lock_fraction: fraction of total equity to move into locked bucket
         risk_mode: new per-trade risk %
         hedge_ratio: proportion of trade hedged
        """

        if lock_fraction > 0:
            lock_amt = self.equity * lock_fraction
            self.locked_profit += lock_amt
            self.equity -= lock_amt
            LOG.info(f"MPC | Locked {lock_amt:.2f}. New equity={self.equity:.2f}, locked={self.locked_profit:.2f}.")

        self.current_risk_mode = risk_mode
        self.current_hedge_ratio = hedge_ratio
        self.free_capital = self.equity

        LOG.info(f"MPC | Updated: risk={risk_mode:.3f}, hedge={hedge_ratio:.2f}, free_capital={self.free_capital:.2f}")

    # ---------------------------------------------------------------
    def create_trade(self, trade_id: str, price: float, stop: float, side: str, now: datetime):
        """Create a new paper trade."""
        risk_amount = self.free_capital * self.current_risk_mode
        stop_dist = abs(price - stop)

        if stop_dist <= 0:
            LOG.error("Stop distance zero — invalid trade.")
            return None

        size = risk_amount / stop_dist

        trade = TradeState(
            trade_id=trade_id,
            side=side,
            entry_price=price,
            stop_price=stop,
            size=size,
            timestamp=now,
            risk_perc=self.current_risk_mode,
        )

        self.open_trades[trade_id] = trade
        LOG.info(f"Trade opened | id={trade_id} side={side} entry={price:.2f} stop={stop:.2f} size={size:.3f}")
        return trade

    # ---------------------------------------------------------------
    def update_trade_r(self, trade: TradeState, price: float):
        """Update R-multiple for an open trade."""
        r = (price - trade.entry_price) / (trade.entry_price - trade.stop_price)
        if trade.side == "short":
            r = -r

        trade.r_mult = r
        trade.highest_r = max(trade.highest_r, r)

    # ---------------------------------------------------------------
    def close_trade(self, trade_id: str, price: float):
        """Close a paper trade and update equity."""
        if trade_id not in self.open_trades:
            return

        trade = self.open_trades[trade_id]
        self.update_trade_r(trade, price)
        pnl = trade.size * (price - trade.entry_price)
        if trade.side == "short":
            pnl = -pnl

        self.equity += pnl
        self.free_capital = self.equity

        LOG.info(f"Trade closed | id={trade_id} pnl={pnl:.2f} equity={self.equity:.2f} r_mult={trade.r_mult:.2f}")
        del self.open_trades[trade_id]

    # ---------------------------------------------------------------
    def close_trade_due_to_stop(self, trade_id: str):
        """Stop-out logic."""
        if trade_id not in self.open_trades:
            return
        trade = self.open_trades[trade_id]
        stop_price = trade.stop_price
        self.close_trade(trade_id, stop_price)
        LOG.info(f"Stopped out | id={trade_id}")

    # ---------------------------------------------------------------
    def apply_price_update(self, price: float):
        """Recompute R for all trades each tick."""
        for t in list(self.open_trades.values()):
            self.update_trade_r(t, price)






"""
forward_state.py

Tracks full forward-test account state:
 - equity, free capital, locked profit
 - open trades
 - trade history (closed trades)
 - rolling equity time series
 - drawdown stats
 - risk mode & hedge ratio
 - cooling regime
"""

from datetime import datetime
from typing import Dict, Any


class ForwardTrade:
    def __init__(self, trade_id, side, entry_price, size, stop):
        self.trade_id = trade_id
        self.side = side
        self.entry_price = entry_price
        self.size = size
        self.stop_price = stop
        self.timestamp = datetime.utcnow()

        # live fields
        self.r_mult = 0.0
        self.highest_r = 0.0


class ForwardState:
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital

        self.equity = starting_capital
        self.free_capital = starting_capital
        self.locked_profit = 0.0

        self.open_trades: Dict[str, ForwardTrade] = {}
        self.closed_trades: Dict[str, Dict[str, Any]] = {}

        self.equity_curve = []          # [(ts, equity)]
        self.max_equity_seen = starting_capital
        self.max_drawdown = 0.0
        self.last_update_ts = None

        self.current_risk_mode = "normal"
        self.current_hedge_ratio = 0.0

        self.cooling_end_time = None    # datetime or None

    # ---------------------------------------------------------
    # CREATE TRADE
    # ---------------------------------------------------------
    def make_trade(self, trade_id, side, entry_price, size, stop):
        t = ForwardTrade(trade_id, side, entry_price, size, stop)
        self.free_capital = max(0.0, self.free_capital - size)
        return t

    # ---------------------------------------------------------
    # CLOSE TRADE
    # ---------------------------------------------------------
    def close_trade(self, trade: ForwardTrade, exit_price: float):

        pnl = (exit_price - trade.entry_price) * (1 if trade.side == "long" else -1)
        r_mult = pnl / (trade.entry_price - trade.stop_price)

        self.free_capital += trade.size + pnl
        self.equity = self.free_capital + self.locked_profit

        self.closed_trades[trade.trade_id] = {
            "trade_id": trade.trade_id,
            "side": trade.side,
            "entry": trade.entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "r_mult": r_mult,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._update_equity_stats()

        return pnl, r_mult

    # ---------------------------------------------------------
    # LOCK PROFIT (MPC)
    # ---------------------------------------------------------
    def apply_profit_lock(self, lock_frac: float):

        to_lock = self.free_capital * lock_frac
        self.locked_profit += to_lock
        self.free_capital -= to_lock

        self.equity = self.free_capital + self.locked_profit
        self._update_equity_stats()

    # ---------------------------------------------------------
    # TRACK EQUITY & DRAWDOWN
    # ---------------------------------------------------------
    def _update_equity_stats(self):

        now = datetime.utcnow()
        self.equity_curve.append((now.isoformat(), self.equity))

        if self.equity > self.max_equity_seen:
            self.max_equity_seen = self.equity

        dd = self.max_equity_seen - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    # ---------------------------------------------------------
    # SET RISK MODE & HEDGE
    # ---------------------------------------------------------
    def set_risk_state(self, risk_mode: str, hedge_ratio: float):
        self.current_risk_mode = risk_mode
        self.current_hedge_ratio = hedge_ratio

    # ---------------------------------------------------------
    # START COOLING PERIOD
    # ---------------------------------------------------------
    def start_cooling(self, until_time: datetime):
        self.cooling_end_time = until_time

    def in_cooling(self, now: datetime):
        if self.cooling_end_time is None:
            return False
        return now < self.cooling_end_time
