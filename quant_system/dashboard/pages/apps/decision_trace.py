from __future__ import annotations

from typing import Any, Dict, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return None


def _event_frame(context: DashboardContext) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    rows = []
    raw_map: Dict[int, Dict[str, Any]] = {}
    for idx, event in enumerate(context.forward.get("events", []) or []):
        if not isinstance(event, dict):
            continue
        payload = _as_dict(event.get("payload"))
        reasoning = _as_dict(payload.get("reasoning"))
        final = _as_dict(reasoning.get("final_decision"))
        hazard_block = _as_dict(reasoning.get("hazard"))
        regime_block = _as_dict(reasoning.get("regime"))
        flow_block = _as_dict(reasoning.get("flow"))
        conf_breakdown = _as_dict(reasoning.get("confluence_breakdown"))
        evr_pack = _pick(payload.get("evr"), reasoning.get("evr"), final.get("evr"))
        if isinstance(evr_pack, dict):
            evr_value = _pick(evr_pack.get("evr"), evr_pack.get("EVR"))
            median_r = _pick(
                evr_pack.get("median_r"),
                evr_pack.get("median_R"),
                final.get("median_r"),
                payload.get("median_r"),
            )
        else:
            evr_value = evr_pack
            median_r = _pick(final.get("median_r"), payload.get("median_r"))
        gates = _pick(payload.get("gates"), payload.get("gate"), payload.get("gate_result"))
        gate_pass = None
        if isinstance(gates, dict) and "passed" in gates:
            gate_pass = bool(gates.get("passed"))

        rows.append(
            {
                "event_idx": idx,
                "timestamp": _pick(
                    event.get("timestamp"),
                    payload.get("timestamp"),
                    payload.get("dt"),
                    payload.get("entry_ts"),
                    payload.get("exit_ts"),
                ),
                "event_type": str(event.get("event_type") or "unknown"),
                "trade_id": str(event.get("trade_id") or payload.get("trade_id") or ""),
                "asset": _pick(payload.get("asset"), payload.get("symbol")),
                "side": payload.get("side"),
                "leg": _pick(payload.get("leg"), final.get("leg"), payload.get("meta_leg")),
                "tier": _pick(payload.get("tier"), final.get("tier")),
                "reason": _pick(payload.get("reason"), final.get("reason"), event.get("event_type")),
                "source": payload.get("source"),
                "confluence": _pick(payload.get("confluence"), final.get("confluence"), conf_breakdown.get("final_confluence")),
                "evr": evr_value,
                "median_r": median_r,
                "hazard": _pick(
                    payload.get("hazard"),
                    payload.get("hazard_score"),
                    hazard_block.get("hazard_score"),
                    payload.get("hazard_entry"),
                ),
                "pnl": _pick(payload.get("pnl"), payload.get("realized_pnl")),
                "r": _pick(payload.get("r"), payload.get("r_mult"), payload.get("realized_r")),
                "regime_state": _pick(payload.get("regime_state"), regime_block.get("regime_state")),
                "session": _pick(payload.get("session"), reasoning.get("session")),
                "gate_pass": gate_pass,
                "p_bos_cont": _pick(conf_breakdown.get("p_bos_cont"), payload.get("p_bos_cont"), payload.get("prob_bos_cont")),
                "p_flow_1h": _pick(conf_breakdown.get("p_flow_1h"), flow_block.get("p_flow_1h"), payload.get("p_flow_1h")),
                "prob_meta": _pick(conf_breakdown.get("prob_meta"), payload.get("prob_meta")),
                "prob_confluence": _pick(conf_breakdown.get("prob_confluence"), payload.get("prob_confluence")),
                "p_regime_trend": regime_block.get("p_regime_trend"),
                "p_regime_expansion": regime_block.get("p_regime_expansion"),
                "p_regime_collapse": regime_block.get("p_regime_collapse"),
            }
        )
        raw_map[idx] = event

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, raw_map

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    numeric_cols = [
        "confluence",
        "evr",
        "median_r",
        "hazard",
        "pnl",
        "r",
        "p_bos_cont",
        "p_flow_1h",
        "prob_meta",
        "prob_confluence",
        "p_regime_trend",
        "p_regime_expansion",
        "p_regime_collapse",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["trade_id"] = frame["trade_id"].fillna("").astype(str)
    frame["event_label"] = (
        frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("na")
        + " | "
        + frame["event_type"].astype(str)
        + " | "
        + frame["asset"].fillna("na").astype(str)
        + " | "
        + frame["trade_id"].replace("", "-")
    )
    return frame.sort_values("timestamp", na_position="last").reset_index(drop=True), raw_map


def _build_lifecycle(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    scoped = events[events["trade_id"].astype(str).str.len() > 0].copy()
    if scoped.empty:
        return pd.DataFrame()

    records = []
    for trade_id, grp in scoped.groupby("trade_id", dropna=False):
        grp = grp.sort_values("timestamp")
        entries = grp[grp["event_type"].isin(["entry", "reasoning", "alert"])]
        exits = grp[grp["event_type"].isin(["exit", "exit_trade", "closed_trade"])]
        entry = entries.iloc[0] if not entries.empty else grp.iloc[0]
        exit_row = exits.iloc[-1] if not exits.empty else None
        entry_ts = entry.get("timestamp")
        exit_ts = exit_row.get("timestamp") if exit_row is not None else pd.NaT
        duration_min = np.nan
        if pd.notna(entry_ts) and pd.notna(exit_ts):
            duration_min = max((exit_ts - entry_ts).total_seconds() / 60.0, 0.0)
        pnl = exit_row.get("pnl") if exit_row is not None else np.nan
        r_val = exit_row.get("r") if exit_row is not None else np.nan
        records.append(
            {
                "trade_id": trade_id,
                "asset": entry.get("asset"),
                "side": entry.get("side"),
                "tier": entry.get("tier"),
                "leg": entry.get("leg"),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "status": "closed" if exit_row is not None else "open",
                "entry_reason": entry.get("reason"),
                "exit_reason": exit_row.get("reason") if exit_row is not None else "",
                "conf_entry": entry.get("confluence"),
                "evr_entry": entry.get("evr"),
                "hazard_entry": entry.get("hazard"),
                "pnl": pnl,
                "r": r_val,
                "duration_min": duration_min,
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    for col in ["conf_entry", "evr_entry", "hazard_entry", "pnl", "r", "duration_min"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("entry_ts", na_position="last").reset_index(drop=True)


def _apply_filters(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    with st.expander("Trace Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        event_types = sorted(events["event_type"].dropna().astype(str).unique().tolist())
        assets = sorted(events["asset"].dropna().astype(str).unique().tolist())
        tiers = sorted(events["tier"].dropna().astype(str).unique().tolist())
        sessions = sorted(events["session"].dropna().astype(str).unique().tolist())

        selected_event_types = c1.multiselect("Event Types", event_types, default=event_types)
        selected_assets = c2.multiselect("Assets", assets, default=assets)
        selected_tiers = c3.multiselect("Tiers", tiers, default=tiers)
        selected_sessions = c4.multiselect("Sessions", sessions, default=sessions)

        c5, c6, c7, c8 = st.columns(4)
        trade_query = c5.text_input("Trade ID contains", value="").strip()
        only_entry_exit = c6.checkbox("Only entry/exit events", value=False)
        conf_min = float(np.nan_to_num(events["confluence"].min(), nan=0.0))
        conf_max = float(np.nan_to_num(events["confluence"].max(), nan=1.0))
        evr_min = float(np.nan_to_num(events["evr"].min(), nan=0.0))
        evr_max = float(np.nan_to_num(events["evr"].max(), nan=2.0))
        min_conf = c7.slider(
            "Min Confluence",
            min_value=conf_min,
            max_value=max(conf_max, conf_min + 1e-6),
            value=conf_min,
        )
        min_evr = c8.slider(
            "Min EVR",
            min_value=evr_min,
            max_value=max(evr_max, evr_min + 1e-6),
            value=evr_min,
        )

    event_mask = events["event_type"].astype(str).isin(selected_event_types) if selected_event_types else True
    asset_mask = events["asset"].fillna("").astype(str).isin(selected_assets) if selected_assets else True
    tier_mask = events["tier"].fillna("").astype(str).isin(selected_tiers) if selected_tiers else True
    session_mask = events["session"].fillna("").astype(str).isin(selected_sessions) if selected_sessions else True
    mask = (
        event_mask
        & asset_mask
        & tier_mask
        & session_mask
        & (events["confluence"].fillna(-np.inf) >= min_conf)
        & (events["evr"].fillna(-np.inf) >= min_evr)
    )
    if trade_query:
        mask &= events["trade_id"].fillna("").astype(str).str.contains(trade_query, case=False, regex=False)
    if only_entry_exit:
        mask &= events["event_type"].isin(["entry", "exit", "exit_trade", "closed_trade"])
    return events.loc[mask].copy().reset_index(drop=True)


def _pnl_style(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if x > 0:
        return "color:#3ddc97;font-weight:700;"
    if x < 0:
        return "color:#ff6b6b;font-weight:700;"
    return "color:#9aa4af;"


def _render_event_stream(events: pd.DataFrame) -> None:
    section_title("Forward/Live Event Stream", "Alert, entry, and exit records from the active event plane")
    view_cols = [
        "timestamp",
        "event_type",
        "trade_id",
        "asset",
        "side",
        "tier",
        "confluence",
        "evr",
        "hazard",
        "reason",
        "source",
    ]
    table = events[[c for c in view_cols if c in events.columns]].copy()
    st.dataframe(table.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True, height=420)

    counts = events.copy()
    counts["minute"] = counts["timestamp"].dt.floor("15min")
    agg = counts.groupby(["minute", "event_type"], dropna=False).size().reset_index(name="count")
    if not agg.empty:
        chart = (
            alt.Chart(agg)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("minute:T", title="Time"),
                y=alt.Y("count:Q", title="Events"),
                color=alt.Color("event_type:N", title="Event"),
                tooltip=["minute:T", "event_type:N", "count:Q"],
            )
            .properties(height=260, title="Event Throughput")
        )
        st.altair_chart(chart, use_container_width=True)


def _render_lifecycle(events: pd.DataFrame) -> None:
    lifecycle = _build_lifecycle(events)
    section_title("Trade Lifecycle", "Entry trigger to exit outcome with causal breadcrumbs")
    if lifecycle.empty:
        st.info("No trade-linked lifecycle is available yet.")
        return

    closed = lifecycle[lifecycle["status"] == "closed"].copy()
    win_rate = float((closed["pnl"] > 0).mean()) if not closed.empty else 0.0
    total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
    metric_grid(
        [
            {"label": "Trades In Trace", "value": f"{len(lifecycle)}"},
            {"label": "Closed Trades", "value": f"{len(closed)}"},
            {"label": "Win Rate (closed)", "value": f"{win_rate * 100:.2f}%"},
            {"label": "Closed PnL", "value": f"${total_pnl:,.2f}"},
        ]
    )

    styled = lifecycle.style.format(
        {
            "conf_entry": "{:.3f}",
            "evr_entry": "{:.3f}",
            "hazard_entry": "{:.3f}",
            "pnl": "{:,.2f}",
            "r": "{:.3f}",
            "duration_min": "{:.1f}",
        }
    ).map(_pnl_style, subset=["pnl", "r"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)


def _render_decision_deep_dive(events: pd.DataFrame, raw_map: Dict[int, Dict[str, Any]]) -> None:
    section_title("Decision Deep Dive", "Inspect one trigger/entry/exit event with full causal payload")
    if events.empty:
        st.info("No events available.")
        return

    choices = events.sort_values("timestamp", ascending=False)
    choice = st.selectbox(
        "Select Event",
        options=choices["event_idx"].tolist(),
        format_func=lambda idx: choices.loc[choices["event_idx"] == idx, "event_label"].iloc[0],
    )
    row = choices.loc[choices["event_idx"] == choice].iloc[0]
    raw = raw_map.get(int(choice), {})
    payload = _as_dict(raw.get("payload"))
    reasoning = _as_dict(payload.get("reasoning"))
    conf_breakdown = _as_dict(reasoning.get("confluence_breakdown"))
    flow = _as_dict(reasoning.get("flow"))
    regime = _as_dict(reasoning.get("regime"))
    smc = _as_dict(reasoning.get("smc"))
    hazard = _as_dict(reasoning.get("hazard"))
    final = _as_dict(reasoning.get("final_decision"))
    event_label = f"Event {row.get('event_type')}"
    tier_label = f"Tier {row.get('tier') or 'n/a'}"
    trade_label = f"Trade {row.get('trade_id') or '-'}"
    tier_tone = "good" if str(row.get("tier") or "").startswith("A") else "neutral"

    st.markdown(
        f"{status_badge(event_label, 'neutral')} "
        f"{status_badge(tier_label, tier_tone)} "
        f"{status_badge(trade_label, 'neutral')}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Confluence", "value": f"{float(row.get('confluence') or 0.0):.3f}"},
            {"label": "EVR", "value": f"{float(row.get('evr') or 0.0):.3f}"},
            {"label": "Median R", "value": f"{float(row.get('median_r') or 0.0):.3f}"},
            {"label": "Hazard", "value": f"{float(row.get('hazard') or 0.0):.3f}"},
            {"label": "Realized R", "value": f"{float(row.get('r') or 0.0):.3f}" if pd.notna(row.get("r")) else "--"},
            {"label": "PnL", "value": f"${float(row.get('pnl') or 0.0):,.2f}" if pd.notna(row.get("pnl")) else "--"},
        ]
    )

    c1, c2, c3 = st.columns([1.0, 1.0, 1.1])
    with c1:
        section_title("Trigger & Decision", "Why the system considered this executable")
        st.json(
            {
                "timestamp": str(row.get("timestamp")),
                "asset": row.get("asset"),
                "side": row.get("side"),
                "tier": row.get("tier"),
                "reason": row.get("reason"),
                "source": row.get("source"),
                "gate_pass": row.get("gate_pass"),
                "final_decision": final,
            }
        )
        section_title("Confluence Vector", "Specialist probabilities and aggregate score")
        st.json(conf_breakdown if conf_breakdown else {"note": "No explicit confluence breakdown payload"})
    with c2:
        section_title("Flow / Regime Context", "Directional energy and state constraints at decision time")
        st.json(
            {
                "flow": flow,
                "regime": regime,
                "session": reasoning.get("session"),
            }
        )
        section_title("SMC Block", "Structure and liquidity primitives")
        st.json(smc if smc else {"note": "No explicit SMC payload"})
    with c3:
        section_title("Risk / Exit Surface", "Hazard and expectancy mechanics")
        st.json(
            {
                "evr": _pick(payload.get("evr"), reasoning.get("evr")),
                "hazard": hazard if hazard else {"hazard_score": row.get("hazard")},
                "realized": {"pnl": row.get("pnl"), "r": row.get("r")},
            }
        )
        trade_id = str(row.get("trade_id") or "")
        if trade_id:
            chain = events[events["trade_id"] == trade_id].sort_values("timestamp")
            st.markdown("**Trade Event Chain**")
            st.dataframe(
                chain[
                    [c for c in ["timestamp", "event_type", "reason", "confluence", "evr", "hazard", "pnl", "r"] if c in chain.columns]
                ],
                use_container_width=True,
                hide_index=True,
                height=220,
            )

    section_title("Raw Event Payload", "Exact event object from the runtime adapter")
    st.json(raw)


def render_decision_trace(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    events, raw_map = _event_frame(context)
    page_header(
        "Decision Trace",
        "Dedicated explainability console for forward/live alerts: why a trade was triggered, entered, managed, and exited.",
        kicker="Causal Audit",
    )
    if events.empty:
        st.info("No forward/live events captured yet.")
        return

    latest = events.iloc[-1]
    entry_count = int((events["event_type"] == "entry").sum())
    exit_count = int(events["event_type"].isin(["exit", "exit_trade", "closed_trade"]).sum())
    filtered = _apply_filters(events)
    latest_event_badge = f"Latest {latest.get('event_type')}"
    model_badge = f"Model set {model_version}"
    model_tone = "good" if model_version != "unavailable" else "warn"

    st.markdown(
        f"{status_badge(latest_event_badge, 'neutral')} "
        f"{status_badge(model_badge, model_tone)}",
        unsafe_allow_html=True,
    )
    metric_grid(
        [
            {"label": "Events", "value": f"{len(events)}"},
            {"label": "Entries", "value": f"{entry_count}"},
            {"label": "Exits", "value": f"{exit_count}"},
            {"label": "Filtered Events", "value": f"{len(filtered)}"},
            {"label": "Latest Confluence", "value": f"{float(latest.get('confluence') or 0.0):.3f}"},
            {"label": "Latest Hazard", "value": f"{float(latest.get('hazard') or 0.0):.3f}"},
        ]
    )

    tabs = st.tabs(["Alert Stream", "Trade Lifecycle", "Decision Deep Dive"])
    with tabs[0]:
        _render_event_stream(filtered)
    with tabs[1]:
        _render_lifecycle(filtered)
    with tabs[2]:
        _render_decision_deep_dive(filtered, raw_map)
