from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext, normalize_trade_frame
from quant_system.dashboard.ui import metric_grid, page_header, section_title


def _forward_closed_trades(context: DashboardContext) -> pd.DataFrame:
    rows = []
    for event in context.forward["events"]:
        if event.get("event_type") not in {"exit", "exit_trade", "closed_trade"}:
            continue
        payload = event.get("payload", {}) or {}
        rows.append(
            {
                "trade_id": event.get("trade_id"),
                "asset": payload.get("asset"),
                "side": payload.get("side"),
                "entry_ts": payload.get("entry_ts") or payload.get("opened_at") or event.get("timestamp"),
                "exit_ts": payload.get("exit_ts") or event.get("timestamp"),
                "entry_price": payload.get("entry_price"),
                "exit_price": payload.get("exit_price"),
                "pnl": payload.get("pnl", 0.0),
                "r": payload.get("r", payload.get("r_mult", 0.0)),
                "tier": payload.get("tier"),
                "conf": payload.get("conf"),
                "evr": payload.get("evr"),
                "reason": payload.get("reason", event.get("event_type")),
                "leg": payload.get("leg"),
            }
        )
    return normalize_trade_frame(pd.DataFrame(rows))


def _prepare_trades(context: DashboardContext) -> pd.DataFrame:
    backtest_trades = context.backtest["trades"]
    forward_trades = _forward_closed_trades(context)
    all_trades = normalize_trade_frame(pd.concat([backtest_trades, forward_trades], ignore_index=True))
    if all_trades.empty:
        return all_trades

    out = all_trades.copy()
    for col in ["pnl", "r", "conf", "evr", "risk", "size_usd", "entry_price", "exit_price", "stop_price", "hazard_entry"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], errors="coerce")
    out["exit_ts"] = pd.to_datetime(out["exit_ts"], errors="coerce")
    out["duration_min"] = ((out["exit_ts"] - out["entry_ts"]).dt.total_seconds() / 60.0).clip(lower=0.0)
    out["duration_hr"] = out["duration_min"] / 60.0
    out["pnl_pct"] = np.where(
        out["size_usd"].abs() > 1e-9,
        (out["pnl"] / out["size_usd"]) * 100.0,
        np.nan,
    )
    out["rr_realized"] = out["r"]
    out["rr_expected"] = out["evr"]
    out["entry_date"] = out["entry_ts"].dt.date
    out["week"] = out["entry_ts"].dt.to_period("W-MON").astype(str)
    out["month"] = out["entry_ts"].dt.to_period("M").astype(str)
    out["outcome"] = np.where(out["pnl"] > 0, "win", np.where(out["pnl"] < 0, "loss", "flat"))
    out["gate_reasons_txt"] = out["gate_reasons"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x or ""))
    out["gates_txt"] = out["gates"].apply(lambda x: ", ".join(f"{k}:{v}" for k, v in x.items()) if isinstance(x, dict) else str(x or ""))
    return out.sort_values("entry_ts").reset_index(drop=True)


def _filter_values(series: pd.Series) -> list:
    return sorted([str(v) for v in series.dropna().astype(str).unique().tolist()])


def _apply_filters(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        assets = _filter_values(trades["asset"])
        sides = _filter_values(trades["side"])
        tiers = _filter_values(trades["tier"])
        legs = _filter_values(trades["leg"])
        regimes = _filter_values(trades["regime"])
        sessions = _filter_values(trades["session"])

        selected_assets = c1.multiselect("Assets", assets, default=assets)
        selected_sides = c2.multiselect("Sides", sides, default=sides)
        selected_tiers = c3.multiselect("Tiers", tiers, default=tiers)
        selected_legs = c4.multiselect("Legs", legs, default=legs)

        c5, c6, c7 = st.columns(3)
        selected_regimes = c5.multiselect("Regimes", regimes, default=regimes)
        selected_sessions = c6.multiselect("Sessions", sessions, default=sessions)
        only_winners = c7.checkbox("Only winners", value=False)

        valid_ts = trades["entry_ts"].dropna()
        if valid_ts.empty:
            date_range = None
        else:
            d_min = valid_ts.min().date()
            d_max = valid_ts.max().date()
            date_range = st.date_input("Entry Date Range", value=(d_min, d_max), min_value=d_min, max_value=d_max)

        conf_min = float(np.nan_to_num(trades["conf"].min(), nan=0.0))
        conf_max = float(np.nan_to_num(trades["conf"].max(), nan=1.0))
        evr_min = float(np.nan_to_num(trades["evr"].min(), nan=-1.0))
        evr_max = float(np.nan_to_num(trades["evr"].max(), nan=2.0))
        r_min = float(np.nan_to_num(trades["r"].min(), nan=-2.0))
        r_max = float(np.nan_to_num(trades["r"].max(), nan=4.0))
        c8, c9, c10 = st.columns(3)
        conf_floor = c8.slider("Min Confluence", min_value=conf_min, max_value=max(conf_max, conf_min + 1e-6), value=conf_min)
        evr_floor = c9.slider("Min EVR", min_value=evr_min, max_value=max(evr_max, evr_min + 1e-6), value=evr_min)
        r_floor = c10.slider("Min Realized R", min_value=r_min, max_value=max(r_max, r_min + 1e-6), value=r_min)

    mask = (
        trades["asset"].astype(str).isin(selected_assets)
        & trades["side"].astype(str).isin(selected_sides)
        & trades["tier"].astype(str).isin(selected_tiers)
        & trades["leg"].astype(str).isin(selected_legs)
        & trades["regime"].astype(str).isin(selected_regimes)
        & trades["session"].astype(str).isin(selected_sessions)
        & (trades["conf"].fillna(-np.inf) >= conf_floor)
        & (trades["evr"].fillna(-np.inf) >= evr_floor)
        & (trades["r"].fillna(-np.inf) >= r_floor)
    )
    if only_winners:
        mask &= trades["pnl"].fillna(0.0) > 0
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask &= trades["entry_ts"].between(start_ts, end_ts)

    return trades.loc[mask].copy().reset_index(drop=True)


def _period_summary(trades: pd.DataFrame, period_col: str, label_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[label_col, "pnl", "trades", "wins", "losses", "win_rate", "avg_r", "avg_conf", "avg_evr"])
    grouped = (
        trades.groupby(period_col, dropna=False)
        .agg(
            pnl=("pnl", "sum"),
            trades=("trade_id", "count"),
            wins=("pnl", lambda s: int((s > 0).sum())),
            losses=("pnl", lambda s: int((s < 0).sum())),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            avg_r=("r", "mean"),
            avg_conf=("conf", "mean"),
            avg_evr=("evr", "mean"),
        )
        .reset_index()
        .rename(columns={period_col: label_col})
    )
    return grouped.sort_values(label_col).reset_index(drop=True)


def _performance_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "profit_factor": 0.0,
            "avg_duration_min": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
        }

    pnl = trades["pnl"].fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "avg_r": float(trades["r"].fillna(0.0).mean()),
        "median_r": float(trades["r"].fillna(0.0).median()),
        "profit_factor": float(profit_factor),
        "avg_duration_min": float(trades["duration_min"].fillna(0.0).mean()),
        "max_win": float(pnl.max()),
        "max_loss": float(pnl.min()),
    }


def _latest_cycle_pnl(trades: pd.DataFrame) -> dict:
    if trades.empty or trades["entry_ts"].dropna().empty:
        return {"day": 0.0, "week": 0.0, "month": 0.0}

    latest = trades["entry_ts"].max()
    day_start = latest.normalize()
    week_start = (latest - pd.Timedelta(days=int(latest.weekday()))).normalize()
    month_start = latest.replace(day=1).normalize()
    return {
        "day": float(trades.loc[trades["entry_ts"] >= day_start, "pnl"].sum()),
        "week": float(trades.loc[trades["entry_ts"] >= week_start, "pnl"].sum()),
        "month": float(trades.loc[trades["entry_ts"] >= month_start, "pnl"].sum()),
    }


def _pnl_bar(df: pd.DataFrame, x_col: str, title: str) -> None:
    if df.empty:
        st.info(f"No {title.lower()} available.")
        return
    plot = df.copy()
    plot["pnl_color"] = np.where(plot["pnl"] >= 0, "profit", "loss")
    chart = (
        alt.Chart(plot)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x_col}:N", sort=None),
            y=alt.Y("pnl:Q", title="PnL"),
            color=alt.Color("pnl_color:N", scale=alt.Scale(domain=["profit", "loss"], range=["#3ddc97", "#ff6b6b"]), legend=None),
            tooltip=[x_col, "pnl", "trades", "win_rate", "avg_r"],
        )
        .properties(height=240, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def _equity_and_distribution(trades: pd.DataFrame) -> None:
    if trades.empty:
        st.info("No trades available for equity analytics.")
        return

    timeline = trades.sort_values("entry_ts").copy()
    timeline["cum_pnl"] = timeline["pnl"].fillna(0.0).cumsum()
    timeline["equity"] = 20_000.0 + timeline["cum_pnl"]
    line = (
        alt.Chart(timeline)
        .mark_line(color="#6ea8fe", strokeWidth=2.2)
        .encode(x="entry_ts:T", y="equity:Q", tooltip=["entry_ts:T", "equity:Q", "cum_pnl:Q"])
        .properties(height=270, title="Equity Curve (Unified Ledger)")
    )
    st.altair_chart(line, use_container_width=True)

    scatter = (
        alt.Chart(timeline)
        .mark_circle(size=64, opacity=0.72)
        .encode(
            x=alt.X("evr:Q", title="Expected Risk/Reward (EVR)"),
            y=alt.Y("r:Q", title="Realized R"),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["win", "loss", "flat"], range=["#3ddc97", "#ff6b6b", "#9aa4af"])),
            tooltip=["trade_id", "asset", "tier", "conf", "evr", "r", "pnl", "reason"],
        )
        .properties(height=300, title="Expected vs Realized Trade Quality")
    )
    st.altair_chart(scatter, use_container_width=True)


def _group_attribution(trades: pd.DataFrame, by_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[by_col, "trades", "pnl", "win_rate", "avg_r", "avg_conf", "avg_evr"])
    return (
        trades.groupby(by_col, dropna=False)
        .agg(
            trades=("trade_id", "count"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            avg_r=("r", "mean"),
            avg_conf=("conf", "mean"),
            avg_evr=("evr", "mean"),
        )
        .reset_index()
        .sort_values("pnl", ascending=False)
    )


def _decile_attribution(trades: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    frame = trades.dropna(subset=[col]).copy()
    if frame.empty or frame[col].nunique() < 4:
        return pd.DataFrame(columns=[label, "trades", "pnl", "win_rate", "avg_r"])
    q = min(10, frame[col].nunique())
    frame[label] = pd.qcut(frame[col], q=q, labels=False, duplicates="drop") + 1
    out = (
        frame.groupby(label, dropna=False)
        .agg(
            trades=("trade_id", "count"),
            pnl=("pnl", "sum"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            avg_r=("r", "mean"),
        )
        .reset_index()
        .sort_values(label)
    )
    return out


def _pnl_cell_style(value) -> str:
    if pd.isna(value):
        return ""
    try:
        x = float(value)
    except Exception:
        return ""
    if x > 0:
        return "color:#3ddc97;font-weight:700;"
    if x < 0:
        return "color:#ff6b6b;font-weight:700;"
    return "color:#9aa4af;"


def _rate_cell_style(value) -> str:
    if pd.isna(value):
        return ""
    try:
        x = float(value)
    except Exception:
        return ""
    if x >= 0.55:
        return "color:#3ddc97;font-weight:700;"
    if x <= 0.45:
        return "color:#ff6b6b;font-weight:700;"
    return "color:#ffd166;font-weight:700;"


def _styled_df(df: pd.DataFrame, *, pnl_cols: list[str] | None = None, rate_cols: list[str] | None = None, pct_cols: list[str] | None = None) -> pd.io.formats.style.Styler:
    pnl_cols = pnl_cols or []
    rate_cols = rate_cols or []
    pct_cols = pct_cols or []
    fmt = {}
    for col in df.columns:
        if col in {"pnl", "avg_pnl", "max_win", "max_loss", "size_usd", "entry_price", "exit_price", "stop_price"}:
            fmt[col] = "{:,.2f}"
        elif col in {"r", "rr_realized", "rr_expected", "avg_r", "median_r", "evr", "avg_evr", "conf", "avg_conf", "hazard_entry"}:
            fmt[col] = "{:.3f}"
        elif col in {"duration_min", "duration_hr"}:
            fmt[col] = "{:.1f}"
        elif col in pct_cols:
            fmt[col] = "{:.2f}%"
        elif col in rate_cols:
            fmt[col] = "{:.1%}"
    styler = df.style.format(fmt)
    if pnl_cols:
        styler = styler.map(_pnl_cell_style, subset=pnl_cols)
    if rate_cols:
        styler = styler.map(_rate_cell_style, subset=rate_cols)
    return styler


def render_trade_log(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    all_trades = _prepare_trades(context)

    page_header(
        "Trade Performance Intelligence",
        "Institutional-grade trade ledger with PnL, risk/reward diagnostics, and ML attribution across every cycle.",
        kicker="Performance Desk",
    )
    if all_trades.empty:
        st.info("No closed trades available yet.")
        return

    filtered = _apply_filters(all_trades)
    stats = _performance_stats(filtered)
    cycles = _latest_cycle_pnl(filtered)

    metric_grid(
        [
            {"label": "Trades", "value": f"{stats['trades']}"},
            {"label": "Win Rate", "value": f"{stats['win_rate'] * 100:.2f}%"},
            {"label": "Total PnL", "value": f"${stats['total_pnl']:,.2f}"},
            {"label": "Profit Factor", "value": "∞" if np.isinf(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}"},
            {"label": "Avg R", "value": f"{stats['avg_r']:.2f}"},
            {"label": "Median R", "value": f"{stats['median_r']:.2f}"},
            {"label": "Avg Trade PnL", "value": f"${stats['avg_pnl']:,.2f}"},
            {"label": "Avg Hold (min)", "value": f"{stats['avg_duration_min']:.1f}"},
        ]
    )
    cycle_col1, cycle_col2, cycle_col3 = st.columns(3)
    cycle_col1.metric("Latest Day PnL", f"${cycles['day']:,.2f}", delta=f"{cycles['day']:,.2f}")
    cycle_col2.metric("Latest Week PnL", f"${cycles['week']:,.2f}", delta=f"{cycles['week']:,.2f}")
    cycle_col3.metric("Latest Month PnL", f"${cycles['month']:,.2f}", delta=f"{cycles['month']:,.2f}")

    tabs = st.tabs(["Overview", "Daily/Weekly/Monthly", "ML Attribution", "Trade Ledger"])

    with tabs[0]:
        section_title("Equity & Trade-Quality Surface", "Cumulative capital curve and expected-vs-realized R diagnostics")
        _equity_and_distribution(filtered)

        section_title("Edge Distribution", "Per-trade PnL histogram (green wins, red losses)")
        if filtered.empty:
            st.info("No trades for distribution.")
        else:
            hist = (
                alt.Chart(filtered.assign(pnl_color=np.where(filtered["pnl"] >= 0, "profit", "loss")))
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X("pnl:Q", bin=alt.Bin(maxbins=50), title="PnL per trade"),
                    y=alt.Y("count():Q", title="Trades"),
                    color=alt.Color("pnl_color:N", scale=alt.Scale(domain=["profit", "loss"], range=["#3ddc97", "#ff6b6b"]), legend=None),
                )
                .properties(height=220)
            )
            st.altair_chart(hist, use_container_width=True)

    daily = _period_summary(filtered, "entry_date", "date")
    weekly = _period_summary(filtered, "week", "week")
    monthly = _period_summary(filtered, "month", "month")

    with tabs[1]:
        section_title("Daily / Weekly / Monthly PnL", "Cycle decomposition with win-rate and average R")
        c1, c2, c3 = st.columns(3)
        with c1:
            _pnl_bar(daily, "date", "Daily PnL")
        with c2:
            _pnl_bar(weekly, "week", "Weekly PnL")
        with c3:
            _pnl_bar(monthly, "month", "Monthly PnL")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.dataframe(
                _styled_df(daily, pnl_cols=["pnl"], rate_cols=["win_rate"]),
                use_container_width=True,
                hide_index=True,
            )
        with d2:
            st.dataframe(
                _styled_df(weekly, pnl_cols=["pnl"], rate_cols=["win_rate"]),
                use_container_width=True,
                hide_index=True,
            )
        with d3:
            st.dataframe(
                _styled_df(monthly, pnl_cols=["pnl"], rate_cols=["win_rate"]),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        section_title("ML / Context Attribution", "How model-aligned dimensions drive realized outcomes")
        g1, g2 = st.columns(2)
        with g1:
            by_tier = _group_attribution(filtered, "tier")
            st.markdown("**By Tier**")
            st.dataframe(_styled_df(by_tier, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)
            by_regime = _group_attribution(filtered, "regime")
            st.markdown("**By Regime**")
            st.dataframe(_styled_df(by_regime, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)
        with g2:
            by_session = _group_attribution(filtered, "session")
            st.markdown("**By Session**")
            st.dataframe(_styled_df(by_session, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)
            by_asset = _group_attribution(filtered, "asset")
            st.markdown("**By Asset**")
            st.dataframe(_styled_df(by_asset, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)

        dec1, dec2, dec3 = st.columns(3)
        with dec1:
            conf_dec = _decile_attribution(filtered, "conf", "conf_decile")
            st.markdown("**Confluence Deciles**")
            st.dataframe(_styled_df(conf_dec, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)
        with dec2:
            evr_dec = _decile_attribution(filtered, "evr", "evr_decile")
            st.markdown("**EVR Deciles**")
            st.dataframe(_styled_df(evr_dec, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)
        with dec3:
            hazard_dec = _decile_attribution(filtered, "hazard_entry", "hazard_decile")
            st.markdown("**Hazard Deciles**")
            st.dataframe(_styled_df(hazard_dec, pnl_cols=["pnl"], rate_cols=["win_rate"]), use_container_width=True, hide_index=True)

    with tabs[3]:
        section_title("Trade Table", "Every trade with PnL, realized/expected R, risk, sizing, regime/session, and reason lineage")
        row_cap = st.number_input("Rows to display", min_value=100, max_value=20000, value=2000, step=100)
        ledger = filtered.tail(int(row_cap)).copy()
        ledger_cols = [
            "trade_id",
            "entry_ts",
            "exit_ts",
            "duration_min",
            "asset",
            "side",
            "tier",
            "leg",
            "conf",
            "evr",
            "rr_realized",
            "pnl",
            "pnl_pct",
            "risk",
            "size_usd",
            "entry_price",
            "exit_price",
            "stop_price",
            "regime",
            "session",
            "reason",
            "gate_reasons_txt",
            "gates_txt",
        ]
        ledger = ledger[[c for c in ledger_cols if c in ledger.columns]]
        st.dataframe(
            _styled_df(
                ledger,
                pnl_cols=["pnl", "rr_realized"],
                rate_cols=[],
                pct_cols=["pnl_pct"],
            ),
            use_container_width=True,
            hide_index=True,
            height=560,
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Filtered Trade Performance CSV",
            csv_data,
            file_name="trade_performance_intelligence.csv",
            mime="text/csv",
        )
