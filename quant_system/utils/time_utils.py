from __future__ import annotations

import pandas as pd

from quant_system.data.prep.sessions import SessionClassifier as CanonicalSessionClassifier


class SessionClassifier:
    """
    Backward-compatible session wrapper that exposes both the canonical session
    columns and the older aliases expected by some feature/runtime code.
    """

    def __init__(self, conf_dir: str = "quant_system/config"):
        self.impl = CanonicalSessionClassifier(conf_dir)

    def classify(self, ts) -> str:
        sample = pd.DataFrame({"dt": [pd.to_datetime(ts, utc=True)]})
        out = self.apply(sample)
        return str(out["session"].iloc[0])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.impl.classify_dataframe(df.copy())
        out["is_ldn"] = out["session_london"]
        out["is_ny"] = out["session_ny"]
        out["session_off_hours"] = out.get("session_offhours", 0)

        if "session" not in out.columns:
            session = pd.Series("off_hours", index=out.index, dtype="object")
            session = session.mask(out["session_london"].astype(bool), "london")
            session = session.mask(out["session_ny"].astype(bool), "ny")
            session = session.mask(out["session_overlap"].astype(bool), "overlap")
            out["session"] = session
        out["session_name"] = out["session"]
        return out

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.apply(df)
