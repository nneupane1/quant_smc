"""
BacktestMetrics:
Computes basic performance, R stats, PnL aggregates, drawdown, and streaks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from quant_system.utils.logger import get_logger

LOG = get_logger("backtest_metrics")


class BacktestMetrics:
    def __init__(self, trades: pd.DataFrame, starting_equity: Optional[float] = None):
        self.df = trades.sort_values("entry_ts") if trades is not None else pd.DataFrame()
        self.start_equity = starting_equity
        if self.df.empty:
            LOG.warning("BacktestMetrics: no trades provided.")

    # ---------------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        if self.df.empty:
            return {"empty": True}

        d = {}
        d.update(self._basic())
        d.update(self._r_stats())
        d.update(self._pnl_stats())
        d.update(self._drawdown())
        d.update(self._streaks())
        d.update(self._override_stats())
        d.update(self._risk_metrics())

        LOG.info("Backtest metrics computed.")
        return d

    # ---------------------------------------------------------------
    def _basic(self) -> Dict[str, Any]:
        wins = (self.df["pnl"] > 0).sum()
        total = len(self.df)
        win_rate = wins / total if total else 0

        long_mask = self.df["side"] == "long"
        short_mask = self.df["side"] == "short"

        long_wins = (self.df[long_mask]["pnl"] > 0).sum()
        short_wins = (self.df[short_mask]["pnl"] > 0).sum()

        # Per-leg breakdown
        def leg_block(leg: str) -> Dict[str, Any]:
            dfl = self.df[self.df["leg"] == leg]
            if dfl.empty:
                return {
                    f"{leg}_trades": 0,
                    f"{leg}_win_rate": np.nan,
                    f"{leg}_avg_r": np.nan,
                    f"{leg}_median_r": np.nan,
                }
            wins_l = (dfl["pnl"] > 0).sum()
            total_l = len(dfl)
            return {
                f"{leg}_trades": total_l,
                f"{leg}_win_rate": wins_l / total_l if total_l else np.nan,
                f"{leg}_avg_r": dfl["r"].mean() if "r" in dfl else np.nan,
                f"{leg}_median_r": dfl["r"].median() if "r" in dfl else np.nan,
            }

        return {
            "trades": total,
            "win_rate": win_rate,
            "win_rate_long": long_wins / long_mask.sum() if long_mask.any() else np.nan,
            "win_rate_short": short_wins / short_mask.sum() if short_mask.any() else np.nan,
            **leg_block("core"),
            **leg_block("runner"),
            **self._regime_stats(),
        }

    # ---------------------------------------------------------------
    def _r_stats(self) -> Dict[str, Any]:
        r = self.df["r"].dropna()
        if r.empty:
            return {"avg_r": np.nan, "median_r": np.nan, "p95_r": np.nan, "p99_r": np.nan}
        return {
            "avg_r": r.mean(),
            "median_r": r.median(),
            "p95_r": r.quantile(0.95),
            "p99_r": r.quantile(0.99),
        }

    # ---------------------------------------------------------------
    def _pnl_stats(self) -> Dict[str, Any]:
        df = self.df.dropna(subset=["exit_ts"])
        if df.empty:
            return {"daily_pnl": {}, "weekly_pnl": {}, "monthly_pnl": {}, "expectancy": np.nan, "profit_factor": np.nan}

        daily = df.groupby(df["exit_ts"].dt.date)["pnl"].sum()
        weekly = df.groupby(df["exit_ts"].dt.to_period("W"))["pnl"].sum()
        monthly = df.groupby(df["exit_ts"].dt.to_period("M"))["pnl"].sum()

        exp = df["pnl"].mean()
        pf = self._profit_factor(df)

        return {
            "daily_pnl": daily.to_dict(),
            "weekly_pnl": weekly.to_dict(),
            "monthly_pnl": monthly.to_dict(),
            "expectancy": exp,
            "profit_factor": pf,
            "cagr": self._cagr(df),
        }

    # ---------------------------------------------------------------
    def _profit_factor(self, df: pd.DataFrame):
        g = df[df["pnl"] > 0]["pnl"].sum()
        l = -df[df["pnl"] < 0]["pnl"].sum()
        return g / l if l > 0 else np.nan

    # ---------------------------------------------------------------
    def _cagr(self, df: pd.DataFrame):
        if df.empty:
            return np.nan
        start = df["entry_ts"].min()
        end = df["exit_ts"].max()
        years = max((end - start).days / 365.0, 1e-9)
        total_return = df["pnl"].sum()
        start_cap = self.start_equity or (df["size_usd"].iloc[0] if "size_usd" in df else 1.0)
        final_cap = start_cap + total_return
        return (final_cap / start_cap) ** (1 / years) - 1 if start_cap > 0 else np.nan

    # ---------------------------------------------------------------
    def _regime_stats(self) -> Dict[str, Any]:
        if "regime" not in self.df:
            return {}
        regimes = self.df["regime"].fillna("unknown")
        out = {}
        for reg in regimes.unique():
            dfr = self.df[regimes == reg]
            if dfr.empty:
                continue
            out[f"win_rate_{reg}"] = (dfr["pnl"] > 0).mean()
            out[f"avg_r_{reg}"] = dfr["r"].mean()
            out[f"trades_{reg}"] = len(dfr)
        return out

    # ---------------------------------------------------------------
    def _drawdown(self):
        base = self.start_equity
        if base is None:
            base = float(self.df["size_usd"].iloc[0]) if "size_usd" in self.df else 0.0
        eq = base + self.df["pnl"].fillna(0).cumsum()
        peak = eq.cummax()
        dd = eq - peak

        return {
            "max_drawdown": dd.min(),
            "drawdown_series": dd.to_list(),
            "equity_curve": eq.to_list(),
        }

    # ---------------------------------------------------------------
    def _streaks(self) -> Dict[str, Any]:
        pnl = self.df["pnl"].fillna(0).values
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for p in pnl:
            if p > 0:
                streak = streak + 1 if streak >= 0 else 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = streak - 1 if streak <= 0 else -1
                max_loss_streak = min(max_loss_streak, streak)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": abs(max_loss_streak),
        }

    # ---------------------------------------------------------------
    def _override_stats(self) -> Dict[str, Any]:
        moon = self.df[self.df["override"] == "moonshot"].shape[0]
        cont = self.df[self.df["override"] == "2R"].shape[0]
        normal = self.df[self.df["override"].isna()].shape[0]

        return {
            "moonshot_overrides": moon,
            "twoR_overrides": cont,
            "normal_entries": normal,
        }

    # ---------------------------------------------------------------
    def _risk_metrics(self) -> Dict[str, Any]:
        if self.df.empty:
            return {}
        base = self.start_equity or (float(self.df["size_usd"].iloc[0]) if "size_usd" in self.df else 1.0)
        # daily returns based on exit pnl
        df = self.df.dropna(subset=["exit_ts"]).copy()
        if df.empty or base == 0:
            return {"sharpe": np.nan, "sortino": np.nan}

        daily = df.groupby(df["exit_ts"].dt.date)["pnl"].sum() / base
        if len(daily) < 2:
            return {"sharpe": np.nan, "sortino": np.nan}

        mean_ret = daily.mean()
        std_ret = daily.std()
        downside = daily[daily < 0].std()

        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret and std_ret > 0 else np.nan
        sortino = mean_ret / downside * np.sqrt(252) if downside and downside > 0 else np.nan

        return {"sharpe": sharpe, "sortino": sortino}
