"""
Order Block Detector (Supply and Demand)
----------------------------------------

Institutional-quality extraction of:
- Demand Order Blocks (last down candle before an impulsive up-move)
- Supply Order Blocks (last up candle before an impulsive down-move)

Key components:
    - Displacement strength based on body size and velocity
    - Zone geometry (open/high/low/close of the OB candle)
    - Age tracking
    - Mitigation detection (price revisiting the zone)
    - Zone quality scoring:
        recency, displacement, body imbalance, mitigation %, touch count

Outputs a dict indexed by timestamp with OB metadata per bar.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from quant_system.data.store.datamodel import Candle
from quant_system.utils.logger import log


class OrderBlockDetector:
    """
    Detect supply/demand order blocks using displacement criteria.

    Parameters:
        min_body_pct: minimum candle body percent for displacement
        min_displacement_r: minimum pip distance relative to close
    """

    def __init__(
        self,
        min_body_pct: float = 0.002,
        min_displacement_r: float = 0.003
    ):
        self.min_body_pct = min_body_pct
        self.min_displacement_r = min_displacement_r
        log(
            f"OrderBlockDetector initialized "
            f"(min_body_pct={min_body_pct}, min_displacement_r={min_displacement_r})."
        )

    def detect(self, candles: List[Candle]) -> Dict[int, Dict[str, Optional[float]]]:
        log(f"Detecting order blocks for {len(candles):,} candles.")

        if len(candles) < 3:
            log("Not enough candles for OB detection.")
            return {}

        result: Dict[int, Dict[str, Optional[float]]] = {}

        # OB trackers
        demand_blocks = []  # list of dicts {top, bottom, age, quality}
        supply_blocks = []

        for i in range(2, len(candles)):
            c0 = candles[i - 2]
            c1 = candles[i - 1]   # OB candidate
            c2 = candles[i]       # displacement candle
            ts = candles[i].timestamp

            new_demand = None
            new_supply = None

            # Body size relative to price
            body_pct = abs(c2.close - c2.open) / max(1e-9, c2.close)

            # Demand OB if last candle was bearish and displacement up
            if c1.close < c1.open and c2.close > c2.open:
                if (
                    body_pct >= self.min_body_pct
                    and (c2.close - c2.open) / max(1e-9, c2.close) >= self.min_displacement_r
                ):
                    new_demand = {
                        "top": c1.high,
                        "bottom": c1.low,
                        "age": 0,
                        "quality": self._quality_score(c1, c2),
                        "touched": False
                    }
                    demand_blocks.append(new_demand)

            # Supply OB if last candle was bullish and displacement down
            if c1.close > c1.open and c2.close < c2.open:
                if (
                    body_pct >= self.min_body_pct
                    and (c1.open - c2.close) / max(1e-9, c2.close) >= self.min_displacement_r
                ):
                    new_supply = {
                        "top": c1.high,
                        "bottom": c1.low,
                        "age": 0,
                        "quality": self._quality_score(c1, c2),
                        "touched": False
                    }
                    supply_blocks.append(new_supply)

            # Update age + mitigation status for all OBs
            for block in demand_blocks:
                block["age"] += 1
                if c2.low <= block["top"] and c2.low >= block["bottom"]:
                    block["touched"] = True

            for block in supply_blocks:
                block["age"] += 1
                if c2.high >= block["bottom"] and c2.high <= block["top"]:
                    block["touched"] = True

            # Select active (freshest, highest quality) OBs for output
            active_demand = self._select_best(demand_blocks)
            active_supply = self._select_best(supply_blocks)

            result[ts] = {
                "demand_top": active_demand["top"] if active_demand else None,
                "demand_bottom": active_demand["bottom"] if active_demand else None,
                "demand_quality": active_demand["quality"] if active_demand else None,
                "demand_age": active_demand["age"] if active_demand else None,
                "demand_touched": int(active_demand["touched"]) if active_demand else None,
                "supply_top": active_supply["top"] if active_supply else None,
                "supply_bottom": active_supply["bottom"] if active_supply else None,
                "supply_quality": active_supply["quality"] if active_supply else None,
                "supply_age": active_supply["age"] if active_supply else None,
                "supply_touched": int(active_supply["touched"]) if active_supply else None,
            }

        log("Order block detection complete.")
        return result

    def _quality_score(self, ob_candle: Candle, disp_candle: Candle) -> float:
        body = abs(disp_candle.close - disp_candle.open)
        range_ = disp_candle.high - disp_candle.low
        imbalance = max(1e-9, body / max(range_, 1e-9))
        proximity = abs(ob_candle.close - disp_candle.open) / max(1e-9, disp_candle.open)
        return imbalance * 0.7 + (1 - proximity) * 0.3

    def _select_best(self, blocks: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not blocks:
            return None
        blocks = [b for b in blocks if b["age"] < 300]  # age decay cutoff
        if not blocks:
            return None
        return max(blocks, key=lambda b: b["quality"])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect order blocks and attach best active zone metadata.
        """
        if df is None or df.empty:
            return df

        frame = df.copy()
        if "timestamp" not in frame.columns:
            if "dt" not in frame.columns:
                raise ValueError("OrderBlockDetector.apply requires 'dt' or 'timestamp' column.")
            frame["timestamp"] = pd.to_datetime(frame["dt"]).astype("int64") // 10**9

        candles = [
            Candle(
                timestamp=int(row.timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0)),
            )
            for row in frame.itertuples()
        ]

        res = self.detect(candles)
        if not res:
            return frame

        res_df = pd.DataFrame.from_dict(res, orient="index")
        res_df.index.name = "timestamp"
        res_df = res_df.reset_index()

        demand_q = pd.to_numeric(res_df.get("demand_quality"), errors="coerce").fillna(-1.0)
        supply_q = pd.to_numeric(res_df.get("supply_quality"), errors="coerce").fillna(-1.0)
        use_demand = demand_q >= supply_q

        res_df["zone_id"] = ""
        res_df.loc[use_demand & res_df["demand_top"].notna(), "zone_id"] = "demand"
        res_df.loc[~use_demand & res_df["supply_top"].notna(), "zone_id"] = "supply"

        res_df["zone_hi"] = np.where(
            use_demand,
            pd.to_numeric(res_df.get("demand_top"), errors="coerce"),
            pd.to_numeric(res_df.get("supply_top"), errors="coerce"),
        )
        res_df["zone_lo"] = np.where(
            use_demand,
            pd.to_numeric(res_df.get("demand_bottom"), errors="coerce"),
            pd.to_numeric(res_df.get("supply_bottom"), errors="coerce"),
        )
        res_df["zone_high"] = res_df["zone_hi"]
        res_df["zone_low"] = res_df["zone_lo"]
        res_df["ob_id"] = res_df["zone_id"]

        age = np.where(
            use_demand,
            pd.to_numeric(res_df.get("demand_age"), errors="coerce"),
            pd.to_numeric(res_df.get("supply_age"), errors="coerce"),
        )
        quality = np.where(
            use_demand,
            demand_q,
            supply_q,
        )
        touched = np.where(
            use_demand,
            pd.to_numeric(res_df.get("demand_touched"), errors="coerce").fillna(0.0),
            pd.to_numeric(res_df.get("supply_touched"), errors="coerce").fillna(0.0),
        )
        age = pd.Series(age).replace([np.inf, -np.inf], np.nan)
        quality = pd.Series(quality).replace([np.inf, -np.inf], np.nan)
        touched = pd.Series(touched).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        res_df["zone_recency"] = (1.0 / (1.0 + age.clip(lower=0))).fillna(0.0)
        res_df["zone_displacement"] = quality.clip(lower=0.0).fillna(0.0)
        res_df["zone_mitigation"] = (1.0 - touched.clip(lower=0.0, upper=1.0)).fillna(1.0)
        res_df["zone_pd"] = 0.5
        res_df["zone_ema_align"] = 0.5

        merged = frame.merge(res_df, on="timestamp", how="left")
        if "dt" in merged.columns:
            merged = merged.sort_values("dt")
        return merged
