from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from quant_system.strategy.types import Position, Leg, Side
from quant_system.strategy.utils import tokens_for_risk, r_multiple, throttle_by_absorption


@dataclass
class PyramidingConfig:
    base_risk_pct: float = 1.0
    add_unit_risk_pct: float = 0.5
    max_live_risk_pct: float = 2.0
    max_adds: int = 3
    absorption_veto: float = 0.70
    absorption_band: tuple = (0.40, 0.70)
    confluence_min: float = 0.60
    session_min_weight: float = 0.20
    be_after_add1: bool = True
    stop_to_pivot_after_each_add: bool = True
    min_stop_buffer_atr: float = 0.2
    partial_r_threshold: float = 2.0
    partial_pct: float = 0.30
    partial_once: bool = True
    allow_regimes: Optional[List[str]] = None
    block_regimes: Optional[List[str]] = None
    rl_allow_override: bool = True


class Pyramider:
    """
    Stateless policy operating on the current row (15m bar) + context hints.
    Call:
      - plan_entry(...) to create initial leg/position
      - maybe_add(...) each bar to decide add-on legs
      - manage_stops_and_partials(...) to tighten/partial
    """

    def __init__(self, cfg: PyramidingConfig):
        self.cfg = cfg

    # ---------- Helpers ----------
    def _gate_quality(self, row: Dict[str, Any]) -> bool:
        if float(row.get("session_weight", 0.0)) < self.cfg.session_min_weight:
            return False
        if float(row.get("confluence_score", 0.0)) < self.cfg.confluence_min:
            return False
        if float(row.get("absorption_score", 0.0)) >= self.cfg.absorption_veto:
            return False
        regime = str(row.get("regime_state", ""))
        if self.cfg.allow_regimes and regime and regime not in self.cfg.allow_regimes:
            return False
        if self.cfg.block_regimes and regime and regime in self.cfg.block_regimes:
            return False
        return True

    def _structural_stop(self, side: Side, row: Dict[str, Any], atr: float) -> Optional[float]:
        buf = self.cfg.min_stop_buffer_atr * atr if atr and atr > 0 else 0.0
        if side == Side.LONG:
            base = row.get("recent_pivot_low") or row.get("ob_low") or row.get("low")
            return float(base) - buf if base is not None else None
        else:
            base = row.get("recent_pivot_high") or row.get("ob_high") or row.get("high")
            return float(base) + buf if base is not None else None

    # ---------- Entry ----------
    def plan_entry(self, equity: float, side: Side, price: float, row: Dict[str, Any]) -> Optional[Position]:
        if not self._gate_quality(row):
            return None

        atr = float(row.get("atr", 0.0))
        stop = self._structural_stop(side, row, atr)
        if not stop:
            return None

        absorption = float(row.get("absorption_score", 0.0))
        mult = throttle_by_absorption(absorption, *self.cfg.absorption_band)
        qty = tokens_for_risk(equity, self.cfg.base_risk_pct * mult, price, stop, side.value)
        if qty <= 0:
            return None

        leg = Leg(
            ts=str(row.get("dt")),
            side=side,
            qty=qty,
            entry_price=price,
            risk_price=stop,
            tokens_risked=equity * (self.cfg.base_risk_pct / 100.0),
            comment="entry",
        )
        pos = Position(id=str(row.get("dt")), side=side, legs=[leg], base_stop=stop)
        pos.recalc()
        return pos

    # ---------- Adds ----------
    def maybe_add(self, pos: Position, equity: float, price: float, row: Dict[str, Any]) -> Optional[Leg]:
        if len(pos.legs) >= self.cfg.max_adds:
            return None
        if not self._gate_quality(row):
            return None

        # structural triggers
        if len(pos.legs) == 1:
            need_ok = (
                bool(row.get("bos_flag_1h", 0))
                and bool(row.get("displacement_15m", 0))
                and bool(row.get("retest_fvg_ob_15m", 0))
            )
            if not need_ok:
                return None
        elif len(pos.legs) == 2:
            need_ok = bool(row.get("bos_flag", 0)) and bool(row.get("fresh_retest_15m", 0))
            if not need_ok:
                return None
        elif len(pos.legs) == 3:
            need_ok = bool(row.get("bos_flag_1h", 0))
            if not need_ok:
                return None

        live_risk_pct = self.cfg.base_risk_pct + (len(pos.legs) - 1) * self.cfg.add_unit_risk_pct
        if live_risk_pct + self.cfg.add_unit_risk_pct > self.cfg.max_live_risk_pct:
            return None

        absorption = float(row.get("absorption_score", 0.0))
        mult = throttle_by_absorption(absorption, *self.cfg.absorption_band)

        atr = float(row.get("atr", 0.0))
        stop = self._structural_stop(pos.side, row, atr)
        if not stop:
            return None
        qty = tokens_for_risk(equity, self.cfg.add_unit_risk_pct * mult, price, stop, pos.side.value)
        if qty <= 0:
            return None

        leg = Leg(
            ts=str(row.get("dt")),
            side=pos.side,
            qty=qty,
            entry_price=price,
            risk_price=stop,
            tokens_risked=equity * (self.cfg.add_unit_risk_pct / 100.0),
            comment=f"add#{len(pos.legs)}",
        )

        if self.cfg.be_after_add1 and len(pos.legs) == 1 and pos.avg_entry:
            pos.base_stop = pos.avg_entry
        if self.cfg.stop_to_pivot_after_each_add:
            pos.base_stop = stop

        pos.legs.append(leg)
        pos.recalc()
        pos.last_add_r_multiple = r_multiple(pos.avg_entry or price, pos.base_stop or price, price, pos.side.value)
        return leg

    # ---------- Ongoing management ----------
    def manage_stops_and_partials(self, pos: Position, price: float, row: Dict[str, Any]) -> Dict[str, Any]:
        out = {"move_stop_to": None, "do_partial": 0.0}
        atr = float(row.get("atr", 0.0))

        if bool(row.get("absorption_near_stop", 0)):
            if pos.side == Side.LONG and pos.base_stop:
                out["move_stop_to"] = max(pos.base_stop, price - 0.5 * atr)
            elif pos.side == Side.SHORT and pos.base_stop:
                out["move_stop_to"] = min(pos.base_stop, price + 0.5 * atr)

        r_now = r_multiple(pos.avg_entry or price, pos.base_stop or price, price, pos.side.value)
        if (r_now >= self.cfg.partial_r_threshold) and (not pos.partial_taken or not self.cfg.partial_once):
            out["do_partial"] = self.cfg.partial_pct
            pos.partial_taken = True

        return out
