"""Session classifier for each timestamp with flexible timestamp sourcing."""

from datetime import datetime, time
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
        self.pre_expansion_minutes = int(self.cfg.get("pre_expansion_minutes", 60))
        self.london_open_expansion_minutes = int(self.cfg.get("london_open_expansion_minutes", 120))
        self.ny_open_expansion_minutes = int(self.cfg.get("ny_open_expansion_minutes", 90))
        self.bucket_quality = self.cfg.get(
            "bucket_quality",
            {
                "dead_zone": 0.75,
                "pre_expansion": 0.95,
                "expansion": 1.05,
                "overlap": 1.15,
            },
        )

        LOG.info("SessionClassifier initialized with London + NY sessions only.")

    def _parse_time(self, s: str) -> time:
        h, m = map(int, s.split(":"))
        return time(hour=h, minute=m)

    def _in_window(self, t: time, start: time, end: time) -> bool:
        """True if t ∈ [start, end)."""
        return start <= t < end

    @staticmethod
    def _minutes_until(local_dt: datetime, target: time) -> int:
        tgt = local_dt.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
        return int((tgt - local_dt).total_seconds() // 60)

    @staticmethod
    def _minutes_since(local_dt: datetime, start: time) -> int:
        st = local_dt.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        return int((local_dt - st).total_seconds() // 60)

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
        session_name = []
        session_bucket = []
        session_bucket_id = []
        session_quality = []
        session_pre_expansion = []
        session_expansion = []
        mins_since_london_open = []
        mins_since_ny_open = []
        mins_to_london_open = []
        mins_to_ny_open = []
        mins_to_overlap_close = []
        overlap_start = max(self.london_start, self.ny_start)
        overlap_close = min(self.london_end, self.ny_end)

        for local_dt, t in zip(local_times, local_tod):
            in_lon = self._in_window(t, self.london_start, self.london_end)
            in_ny = self._in_window(t, self.ny_start, self.ny_end)
            in_overlap = in_lon and in_ny

            if in_overlap:
                w = self.overlap_weight
                sname = "overlap"
            elif in_lon:
                w = self.london_weight
                sname = "london"
            elif in_ny:
                w = self.ny_weight
                sname = "ny"
            else:
                w = self.off_weight
                sname = "off_hours"

            since_lon = self._minutes_since(local_dt, self.london_start) if in_lon else -1
            since_ny = self._minutes_since(local_dt, self.ny_start) if in_ny else -1
            to_lon = self._minutes_until(local_dt, self.london_start)
            to_ny = self._minutes_until(local_dt, self.ny_start)
            to_overlap_close = self._minutes_until(local_dt, overlap_close) if in_overlap else -1

            pre_lon = (not in_lon) and (0 <= to_lon <= self.pre_expansion_minutes)
            pre_ny = (not in_ny) and (0 <= to_ny <= self.pre_expansion_minutes)
            pre = bool(pre_lon or pre_ny)
            expansion = bool(
                (in_lon and 0 <= since_lon <= self.london_open_expansion_minutes)
                or (in_ny and 0 <= since_ny <= self.ny_open_expansion_minutes)
            )
            if in_overlap:
                bucket = "overlap"
                bucket_id = 3
            elif expansion:
                bucket = "expansion"
                bucket_id = 2
            elif pre:
                bucket = "pre_expansion"
                bucket_id = 1
            else:
                bucket = "dead_zone"
                bucket_id = 0
            quality = float(self.bucket_quality.get(bucket, 1.0))

            london.append(int(in_lon))
            ny.append(int(in_ny))
            overlap.append(int(in_overlap))
            offhours.append(int(not (in_lon or in_ny)))
            weights.append(w)
            session_name.append(sname)
            session_bucket.append(bucket)
            session_bucket_id.append(bucket_id)
            session_quality.append(quality)
            session_pre_expansion.append(int(pre))
            session_expansion.append(int(expansion))
            mins_since_london_open.append(int(since_lon))
            mins_since_ny_open.append(int(since_ny))
            mins_to_london_open.append(int(to_lon))
            mins_to_ny_open.append(int(to_ny))
            mins_to_overlap_close.append(int(to_overlap_close))

        base_df["session_london"] = london
        base_df["session_ny"] = ny
        base_df["session_overlap"] = overlap
        base_df["session_offhours"] = offhours
        base_df["session_weight"] = weights
        base_df["session"] = session_name
        base_df["session_name"] = session_name
        base_df["session_bucket"] = session_bucket
        base_df["session_bucket_id"] = session_bucket_id
        base_df["session_quality_multiplier"] = session_quality
        base_df["session_pre_expansion"] = session_pre_expansion
        base_df["session_expansion"] = session_expansion
        base_df["minutes_since_london_open"] = mins_since_london_open
        base_df["minutes_since_ny_open"] = mins_since_ny_open
        base_df["minutes_to_london_open"] = mins_to_london_open
        base_df["minutes_to_ny_open"] = mins_to_ny_open
        base_df["minutes_to_overlap_close"] = mins_to_overlap_close

        LOG.info("Session classification complete.")
        return base_df
