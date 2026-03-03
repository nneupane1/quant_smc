from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from quant_system.strategy.types import Position, Leg, Side
from quant_system.strategy.utils import (
    boolish,
    first_number,
    first_text,
    r_multiple,
    throttle_by_absorption,
    tokens_for_risk,
)


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
    def _atr(self, row: Dict[str, Any]) -> float:
        return first_number(row, ["atr_15m", "atr"], 0.0)

    def _confluence(self, row: Dict[str, Any]) -> float:
        return first_number(row, ["confluence_score", "conf_score", "prob_confluence", "conf"], 0.0)

    def _regime(self, row: Dict[str, Any]) -> str:
        return first_text(row, ["regime_state", "regime", "dominant_regime"], "")

    def _flow_ok(self, row: Dict[str, Any]) -> bool:
        if boolish(row.get("bos_flag_1h")):
            return True
        if first_number(row, ["prob_flow_1h", "p_flow_1h", "flow_1h"], 0.0) >= 0.55:
            return True
        return first_number(row, ["flow_strength_1h"], 0.0) > 0.0

    def _displacement_ok(self, row: Dict[str, Any]) -> bool:
        return boolish(row.get("displacement_15m")) or first_number(row, ["body_pct_15m"], 0.0) >= 0.55

    def _retest_ok(self, row: Dict[str, Any]) -> bool:
        return boolish(row.get("fresh_retest_15m")) or boolish(row.get("retest_fvg_ob_15m"))

    def _gate_quality(self, row: Dict[str, Any]) -> bool:
        if float(row.get("session_weight", 0.0)) < self.cfg.session_min_weight:
            return False
        if self._confluence(row) < self.cfg.confluence_min:
            return False
        if first_number(row, ["absorption_score"], 0.0) >= self.cfg.absorption_veto:
            return False
        regime = self._regime(row)
        if self.cfg.allow_regimes and regime and regime not in self.cfg.allow_regimes:
            return False
        if self.cfg.block_regimes and regime and regime in self.cfg.block_regimes:
            return False
        return True

    def _structural_stop(self, side: Side, row: Dict[str, Any], atr: float) -> Optional[float]:
        buf = self.cfg.min_stop_buffer_atr * atr if atr and atr > 0 else 0.0
        if side == Side.LONG:
            base = next(
                (
                    row.get(key)
                    for key in ("recent_pivot_low", "swing_low", "ob_low", "zone_lo", "zone_low", "demand_zone", "low")
                    if row.get(key) is not None
                ),
                None,
            )
            return float(base) - buf if base is not None else None
        else:
            base = next(
                (
                    row.get(key)
                    for key in ("recent_pivot_high", "swing_high", "ob_high", "zone_hi", "zone_high", "supply_zone", "high")
                    if row.get(key) is not None
                ),
                None,
            )
            return float(base) + buf if base is not None else None

    # ---------- Entry ----------
    def plan_entry(self, equity: float, side: Side, price: float, row: Dict[str, Any]) -> Optional[Position]:
        if not self._gate_quality(row):
            return None

        atr = self._atr(row)
        stop = self._structural_stop(side, row, atr)
        if not stop:
            return None

        absorption = first_number(row, ["absorption_score"], 0.0)
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
        add_count = max(len(pos.legs) - 1, 0)
        if add_count >= self.cfg.max_adds:
            return None
        if not self._gate_quality(row):
            return None

        # structural triggers
        if len(pos.legs) == 1:
            need_ok = (
                self._flow_ok(row)
                and self._displacement_ok(row)
                and self._retest_ok(row)
            )
            if not need_ok:
                return None
        elif len(pos.legs) == 2:
            need_ok = boolish(row.get("bos_flag")) and self._retest_ok(row)
            if not need_ok:
                return None
        elif len(pos.legs) == 3:
            need_ok = self._flow_ok(row)
            if not need_ok:
                return None

        live_risk_pct = self.cfg.base_risk_pct + (len(pos.legs) - 1) * self.cfg.add_unit_risk_pct
        if live_risk_pct + self.cfg.add_unit_risk_pct > self.cfg.max_live_risk_pct:
            return None

        absorption = first_number(row, ["absorption_score"], 0.0)
        mult = throttle_by_absorption(absorption, *self.cfg.absorption_band)

        atr = self._atr(row)
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
        atr = self._atr(row)

        if boolish(row.get("absorption_near_stop")):
            if pos.side == Side.LONG and pos.base_stop:
                out["move_stop_to"] = max(pos.base_stop, price - 0.5 * atr)
            elif pos.side == Side.SHORT and pos.base_stop:
                out["move_stop_to"] = min(pos.base_stop, price + 0.5 * atr)

        r_now = r_multiple(pos.avg_entry or price, pos.base_stop or price, price, pos.side.value)
        if (r_now >= self.cfg.partial_r_threshold) and (not pos.partial_taken or not self.cfg.partial_once):
            out["do_partial"] = self.cfg.partial_pct
            pos.partial_taken = True

        return out
