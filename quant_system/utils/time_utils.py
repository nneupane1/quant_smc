import pandas as pd


class SessionClassifier:
    """
    Simple session classifier placeholder.
    """

    def __init__(self):
        # Define session windows if needed
        self.sessions = {
            "asia": (0, 8),
            "london": (8, 16),
            "ny": (13, 21),
        }

    def classify(self, ts) -> str:
        hour = pd.to_datetime(ts).hour if ts is not None else 0
        for name, (start, end) in self.sessions.items():
            if start <= hour < end:
                return name
        return "other"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'session' column based on timestamp/dt.
        """
        df = df.copy()
        if "dt" in df.columns:
            df["session"] = df["dt"].apply(self.classify)
        elif "timestamp" in df.columns:
            df["session"] = pd.to_datetime(df["timestamp"], unit="s").apply(self.classify)
        else:
            df["session"] = "other"
        return df
