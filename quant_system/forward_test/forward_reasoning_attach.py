"""
Minimal reasoning attachment for forward-test events.
"""

from typing import Any, Dict


class ReasoningAttach:
    @staticmethod
    def _pick(row: Dict[str, Any], *keys):
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    def build(
        self,
        asset: str,
        row: Dict[str, Any],
        conf: float,
        evr: Dict[str, Any],
        tier: str,
        risk_mode: Any,
        hedge_ratio: float,
    ) -> Dict[str, Any]:
        confluence_breakdown = {
            "p_liq_flow": row.get("p_liq_flow", row.get("prob_liq_flow")),
            "p_bos_cont": row.get("p_bos_cont", row.get("prob_bos_cont")),
            "p_flow_1h": row.get("p_flow_1h", row.get("prob_flow_1h")),
            "p_momo": row.get("p_momo", row.get("prob_momo")),
            "prob_meta": row.get("prob_meta"),
            "prob_confluence": row.get("prob_confluence"),
            "final_confluence": conf,
        }

        return {
            "asset": asset,
            "tier": tier,
            "confluence": conf,
            "evr": evr,
            "risk_mode": risk_mode,
            "hedge_ratio": hedge_ratio,
            "reasoning": {
                "ml": {k: v for k, v in row.items() if str(k).startswith(("p_", "prob_"))},
                "smc": {
                    "swings": {
                        "swing_high": self._pick(row, "swing_high", "swing_high_15m"),
                        "swing_low": self._pick(row, "swing_low", "swing_low_15m"),
                    },
                    "bos": {
                        "bos_up": self._pick(row, "bos_up", "bos_flag"),
                        "bos_down": row.get("bos_down"),
                        "choch": self._pick(row, "choch_flag", "choch_flag_1h", "choch_flag_6h"),
                    },
                    "sweeps": {
                        "sweep_high": row.get("sweep_high"),
                        "sweep_low": row.get("sweep_low"),
                    },
                    "zones": {
                        "zone_score_6h": row.get("zone_score_6h"),
                        "demand_zone": row.get("demand_zone"),
                        "supply_zone": row.get("supply_zone"),
                    },
                },
                "regime": {
                    "regime_state": row.get("regime_state"),
                    "p_regime_trend": row.get("p_regime_trend"),
                    "p_regime_expansion": row.get("p_regime_expansion"),
                    "p_regime_collapse": row.get("p_regime_collapse"),
                    "toxicity_12h": row.get("toxicity_12h"),
                },
                "flow": {
                    "p_flow_1h": row.get("p_flow_1h", row.get("prob_flow_1h")),
                    "flow_signal_1h": row.get("flow_signal_1h"),
                    "flow_strength_1h": row.get("flow_strength_1h"),
                    "flow_age_bars_1h": row.get("flow_age_bars_1h"),
                    "displacement_body_pct_1h": row.get("displacement_body_pct_1h"),
                    "volume_z_1h": row.get("volume_z_1h"),
                },
                "ema": {
                    "ema_fast": self._pick(row, "ema_fast", "ema_fast_15m"),
                    "ema_slow": self._pick(row, "ema_slow", "ema_slow_15m"),
                    "dist_ema": self._pick(row, "dist_ema", "dist_to_ema"),
                    "band_regime": row.get("band_regime"),
                },
                "confluence_breakdown": confluence_breakdown,
                "evr": evr,
                "hazard": {
                    "hazard_score": row.get("hazard_score", row.get("hazard")),
                    "hazard_curve": row.get("hazard_curve"),
                },
                "final_decision": {
                    "tier": tier,
                    "risk_mode": risk_mode,
                    "hedge_ratio": hedge_ratio,
                    "confluence": conf,
                    "evr": evr.get("evr") if isinstance(evr, dict) else evr,
                    "median_r": evr.get("median_r") if isinstance(evr, dict) else None,
                },
                "session": row.get("session"),
            },
        }
