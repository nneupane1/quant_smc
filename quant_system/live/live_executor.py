"""
live_executor.py
Kraken execution wrapper with simple capital bookkeeping and optional hedge leg.
"""

import math
import time
from datetime import datetime
from typing import Optional, Dict, Any

from quant_system.live.kraken_live_client import KrakenLiveClient
from quant_system.config.config_loader import ConfigLoader
from quant_system.utils.logger import get_logger

LOG = get_logger("live_executor")


class LivePosition:
    """Represents an open live position."""

    def __init__(
        self,
        trade_id: str,
        asset: str,
        side: str,
        qty: float,
        entry_price: float,
        leverage: int,
        meta: Dict[str, Any],
        stop_price: Optional[float] = None,
    ):
        self.trade_id = trade_id
        self.asset = asset
        self.side = side              # long / short
        self.qty = qty
        self.entry_price = entry_price
        self.leverage = leverage
        self.meta = meta or {}
        self.stop_price = stop_price
        self.open_time = datetime.utcnow()
        self.bars_in_trade = 0
        self.hedge_leg = None         # assign perp hedge ID later


class LiveExecutor:
    """
    Live execution engine wrapping KrakenLiveClient.
    Handles:
      - open_position
      - exit_position
      - hedge_position
      - mark_to_market (from live quotes)
    """

    def __init__(self, cfg: ConfigLoader, dashboard_adapter=None):
        self.cfg = cfg
        self.exec_cfg = cfg.load_yaml("execution.yaml")
        self.assets_cfg = cfg.load_yaml("assets.yaml")["assets"]["metadata"]

        self.kraken = KrakenLiveClient(cfg)
        self.dashboard = dashboard_adapter
        self.positions: Dict[str, LivePosition] = {}

        self.equity = float(self.exec_cfg.get("starting_equity", 0))
        self.locked_profit = 0.0
        self.free_capital = self.equity

        # leverage mode
        self.leverage_enabled = self.exec_cfg.get("enable_leverage", False)
        self.max_lev = self.exec_cfg.get("max_leverage", 1)

        LOG.info(f"[LiveExecutor] Initialized. Leverage enabled={self.leverage_enabled}, max={self.max_lev}")

    # ----------------------------------------------------------
    # MAIN ENTRY: OPEN POSITION
    # ----------------------------------------------------------
    def open_position(self, trade_id: str, asset: str, usd_size: float, price: float, meta: Dict[str, Any], stop_price: Optional[float] = None):
        """
        Creates a position:
          - Determine leverage
          - Convert USD -> base qty
          - Submit order
          - Return LivePosition
        """

        symbol_cfg = self.assets_cfg[asset]
        min_size = symbol_cfg["min_size"]

        side = "long" if meta.get("direction", "long") == "long" else "short"

        lev = 1
        if self.leverage_enabled and symbol_cfg["leverage_allowed"]:
            lev = max(1, min(self.max_lev, meta.get("leverage", 1)))

        qty = usd_size * lev / price
        qty = self._round(qty, min_size)

        if qty <= 0:
            LOG.warning(f"[LiveExecutor] Qty {qty} invalid. Skipping trade.")
            return None

        LOG.info(f"[LiveExecutor] Opening {side.upper()} {asset} qty={qty} lev={lev} price={price}")

        order_id = self._submit_order(asset, side, qty, price, lev)
        if not order_id:
            LOG.error("[LiveExecutor] Order failed.")
            return None

        pos = LivePosition(
            trade_id=trade_id,
            asset=asset,
            side=side,
            qty=qty,
            entry_price=price,
            leverage=lev,
            meta=meta,
            stop_price=stop_price,
        )
        self.positions[trade_id] = pos

        # Capital bookkeeping (simplified; assumes full notional reserved)
        self.free_capital -= usd_size
        self.equity = self.locked_profit + self.free_capital

        if self.dashboard:
            self.dashboard.log_event("live_entry", trade_id, {
                "asset": asset,
                "qty": qty,
                "side": side,
                "leverage": lev,
                "entry": price,
                "leg": meta.get("leg"),
            })

        return pos

    # ----------------------------------------------------------
    # EXIT POSITION
    # ----------------------------------------------------------
    def exit_position(self, trade_id: str, price: float):
        """
        Market exit the position.
        """

        pos = self.positions.get(trade_id)
        if not pos:
            LOG.error(f"[LiveExecutor] exit: position {trade_id} not found")
            return 0.0

        side = "sell" if pos.side == "long" else "buy"
        LOG.info(f"[LiveExecutor] Exiting {pos.asset} [{trade_id}] qty={pos.qty} at price={price}")

        self._submit_order(pos.asset, side, pos.qty, price, pos.leverage)

        pnl = self._calculate_pnl(pos, price)
        value = pos.qty * price
        self.free_capital += value
        self.equity = self.locked_profit + self.free_capital
        del self.positions[trade_id]

        if self.dashboard:
            self.dashboard.log_event("live_exit", trade_id, {"pnl": pnl, "value": value})

        return pnl

    # ----------------------------------------------------------
    # PERP HEDGE LEG
    # ----------------------------------------------------------
    def hedge_position(self, trade_id: str, hedge_ratio: float, price: float):
        """
        Open or adjust a hedge short for a long position.
        hedge_ratio: 0.0–1.0 (fraction of delta)
        """

        pos = self.positions.get(trade_id)
        if not pos:
            LOG.error("[LiveExecutor] hedge: no position")
            return None

        if pos.side != "long":
            # hedging only for bullish core positions
            return None

        base_qty = pos.qty
        hedge_qty = base_qty * hedge_ratio

        if hedge_qty <= 0:
            return None

        symbol_cfg = self.assets_cfg[pos.asset]
        min_size = symbol_cfg["min_size"]
        hedge_qty = self._round(hedge_qty, min_size)

        LOG.info(f"[LiveExecutor] Hedging {pos.asset}: qty={hedge_qty} at price={price}")

        order_id = self._submit_order(pos.asset, "sell", hedge_qty, price, pos.leverage)
        if order_id:
            pos.hedge_leg = order_id

        return order_id

    # ----------------------------------------------------------
    # MTM
    # ----------------------------------------------------------
    def mark_to_market(self, trade_id: str, price: float):
        pos = self.positions.get(trade_id)
        if not pos:
            return 0.0
        return (price - pos.entry_price) * pos.qty * (1 if pos.side == "long" else -1)

    # ----------------------------------------------------------
    # SUBMIT ORDER (spot + margin mode)
    # ----------------------------------------------------------
    def _submit_order(self, asset: str, side: str, qty: float, price: float, lev: int):
        """
        Call Kraken API safely.
        """

        pair = self.assets_cfg[asset]["kraken_pair"]
        side_norm = "buy" if side in ["buy", "long"] else "sell"

        try:
            resp = self.kraken.submit_order(
                pair=pair,
                side=side_norm,
                volume=qty,
                price=price,
                leverage=lev
            )
            oid = resp.get("txid")
            LOG.info(f"[LiveExecutor] Order success {pair}, oid={oid}")
            return oid
        except Exception as e:
            LOG.error(f"[LiveExecutor] Order failed: {e}")
            return None

    # ----------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------
    def _round(self, v: float, step: float) -> float:
        return math.floor(v / step) * step

    def _calculate_pnl(self, pos: LivePosition, price: float):
        direction = 1 if pos.side == "long" else -1
        return (price - pos.entry_price) * pos.qty * direction
