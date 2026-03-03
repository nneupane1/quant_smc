"""Session classifier for each timestamp with flexible timestamp sourcing."""

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from quant_system.config.config_manager import ConfigManager
from quant_system.utils.logger import get_logger

LOG = get_logger("session_classifier")


class SessionClassifier:
    """
    Classifies each row into trading sessions:
        - London
        - New York
        - Overlap
        - Off-hours
    And assigns session weights for confluence scoring.
    """

    def __init__(self, conf_dir: str):
        self.cfg = ConfigManager(conf_dir).get("features")["session"]
        self.tz = ZoneInfo(self.cfg.get("timezone", "Europe/Berlin"))

        self.london_start = self._parse_time(self.cfg["london"]["start"])
        self.london_end = self._parse_time(self.cfg["london"]["end"])
        self.london_weight = float(self.cfg["london"]["weight"])

        self.ny_start = self._parse_time(self.cfg["ny"]["start"])
        self.ny_end = self._parse_time(self.cfg["ny"]["end"])
        self.ny_weight = float(self.cfg["ny"]["weight"])

        self.overlap_weight = float(self.cfg["overlap"]["weight"])
        self.off_weight = float(self.cfg["off_hours"]["weight"])

        LOG.info("SessionClassifier initialized with London + NY sessions only.")

    def _parse_time(self, s: str) -> time:
        h, m = map(int, s.split(":"))
        return time(hour=h, minute=m)

    def _in_window(self, t: time, start: time, end: time) -> bool:
        """True if t ∈ [start, end)."""
        return start <= t < end

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add session columns + session weight to dataframe.
        Accepts either a timezone-aware datetime index or `dt` / `timestamp` columns.
        """

        if df.empty:
            return df

        LOG.info("Classifying sessions for dataset...")

        base_df = df.copy()
        if isinstance(base_df.index, pd.DatetimeIndex):
            idx = base_df.index
        elif "dt" in base_df.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(base_df["dt"], utc=True))
        elif "timestamp" in base_df.columns:
            ts_series = base_df["timestamp"]
            if pd.api.types.is_numeric_dtype(ts_series):
                numeric_ts = pd.to_numeric(ts_series, errors="coerce")
                unit = "ms" if numeric_ts.dropna().gt(10**11).any() else "s"
                idx = pd.DatetimeIndex(pd.to_datetime(numeric_ts, unit=unit, utc=True))
            else:
                idx = pd.DatetimeIndex(pd.to_datetime(ts_series, utc=True, errors="coerce"))
        else:
            raise ValueError("SessionClassifier requires a DatetimeIndex or `dt` / `timestamp` columns.")

        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        # Convert index timestamps to local session timezone.
        local_times = idx.tz_convert(self.tz)
        local_tod = local_times.time

        london = []
        ny = []
        overlap = []
        offhours = []
        weights = []

        for t in local_tod:
            in_lon = self._in_window(t, self.london_start, self.london_end)
            in_ny = self._in_window(t, self.ny_start, self.ny_end)
            in_overlap = in_lon and in_ny

            if in_overlap:
                w = self.overlap_weight
            elif in_lon:
                w = self.london_weight
            elif in_ny:
                w = self.ny_weight
            else:
                w = self.off_weight

            london.append(int(in_lon))
            ny.append(int(in_ny))
            overlap.append(int(in_overlap))
            offhours.append(int(not (in_lon or in_ny)))
            weights.append(w)

        base_df["session_london"] = london
        base_df["session_ny"] = ny
        base_df["session_overlap"] = overlap
        base_df["session_offhours"] = offhours
        base_df["session_weight"] = weights

        LOG.info("Session classification complete.")
        return base_df
