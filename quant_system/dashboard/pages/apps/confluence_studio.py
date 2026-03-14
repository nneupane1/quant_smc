from __future__ import annotations

from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext, normalize_trade_frame
from quant_system.dashboard.intelligence import candidate_frame, execution_snapshot, latest_market_frame, latest_market_row, latest_reasoning
from quant_system.dashboard.ui import inject_page_notice, metric_grid, page_header, section_title, status_badge


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


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return None if np.isnan(number) else number


def _fmt(value: Any, pattern: str = "{:.3f}", fallback: str = "--") -> str:
    number = _to_float(value)
    if number is None:
        return fallback
    return pattern.format(number)


def _fmt_delta(value: Any, fallback: str = "--") -> str:
    number = _to_float(value)
    if number is None:
        return fallback
    return f"{number:+.3f}"


def _history_frame(context: DashboardContext, limit: int = 240) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for event in context.forward.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        payload = _as_dict(event.get("payload"))
        reasoning = _as_dict(payload.get("reasoning"))
        conf = _as_dict(reasoning.get("confluence_breakdown"))
        final = _as_dict(reasoning.get("final_decision"))
        hazard = _as_dict(reasoning.get("hazard"))
        evr_block = _pick(payload.get("evr"), reasoning.get("evr"), final.get("evr"))
        evr_value = evr_block.get("evr") if isinstance(evr_block, dict) else evr_block
        median_r = evr_block.get("median_r") if isinstance(evr_block, dict) else _pick(payload.get("median_r"), final.get("median_r"))
        rows.append(
            {
                "timestamp": _pick(event.get("timestamp"), payload.get("timestamp"), payload.get("dt")),
                "event_type": event.get("event_type"),
                "trade_id": _pick(event.get("trade_id"), payload.get("trade_id")),
                "tier": _pick(payload.get("tier"), final.get("tier")),
                "final_confluence": _pick(payload.get("confluence"), final.get("confluence"), conf.get("final_confluence")),
                "prob_confluence": _pick(payload.get("prob_confluence"), conf.get("prob_confluence")),
                "prob_meta": _pick(payload.get("prob_meta"), conf.get("prob_meta")),
                "hazard": _pick(payload.get("hazard"), payload.get("hazard_score"), hazard.get("hazard_score")),
                "evr": evr_value,
                "median_r": median_r,
            }
        )
    if rows:
        frame = pd.DataFrame(rows)
    else:
        trades = normalize_trade_frame(context.backtest.get("trades", pd.DataFrame())).copy()
        if trades.empty:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "timestamp": trades["entry_ts"],
                "event_type": trades.get("result", pd.Series(["trade"] * len(trades))),
                "trade_id": trades["trade_id"],
                "tier": trades.get("tier"),
                "final_confluence": pd.to_numeric(trades.get("conf"), errors="coerce"),
                "prob_confluence": np.nan,
                "prob_meta": np.nan,
                "hazard": pd.to_numeric(trades.get("hazard_entry"), errors="coerce"),
                "evr": pd.to_numeric(trades.get("evr"), errors="coerce"),
                "median_r": pd.to_numeric(trades.get("r"), errors="coerce"),
            }
        )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ("final_confluence", "prob_confluence", "prob_meta", "hazard", "evr", "median_r"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").tail(limit).reset_index(drop=True)


def _specialist_surface(row: Dict[str, Any], reasoning: Dict[str, Any]) -> pd.DataFrame:
    conf = _as_dict(reasoning.get("confluence_breakdown"))
    mapping = [
        ("Liquidity Flow", _pick(conf.get("p_liq_flow"), row.get("p_liq_flow"))),
        ("BOS Continuation", _pick(conf.get("p_bos_cont"), row.get("p_bos_cont"))),
        ("Flow 1H", _pick(conf.get("p_flow_1h"), row.get("p_flow_1h"), row.get("prob_flow_1h"))),
        ("Momentum", _pick(conf.get("p_momo"), row.get("p_momo"))),
        ("EOP", _pick(conf.get("p_eop"), row.get("p_eop"))),
        ("EDP", _pick(conf.get("p_edp"), row.get("p_edp"))),
        ("Meta Stack", _pick(conf.get("prob_meta"), row.get("prob_meta"))),
        ("Confluence Stack", _pick(conf.get("prob_confluence"), row.get("prob_confluence"))),
    ]
    out = pd.DataFrame(mapping, columns=["channel", "value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _routing_tables(state: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specialist_source = state.get("specialist_model_source", {})
    stack_source = state.get("stack_model_source", {})
    route_df = pd.DataFrame(
        [
            {"component": key, "model": value}
            for key, value in sorted(specialist_source.items())
        ]
    )
    stack_df = pd.DataFrame(
        [
            {"stack": key, "model": value}
            for key, value in sorted(stack_source.items())
        ]
    )
    return route_df, stack_df


def render_confluence_studio(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    state = context.forward.get("state", {}) or {}
    row = latest_market_row(context)
    market = latest_market_frame(context)
    reasoning = latest_reasoning(context)
    snapshot = execution_snapshot(context)
    history = _history_frame(context)
    candidates = candidate_frame(context)

    confluence_state = _as_dict(snapshot.get("confluence"))
    conf = _as_dict(reasoning.get("confluence_breakdown"))
    final_decision = _as_dict(reasoning.get("final_decision"))

    final_confluence = _pick(row.get("confluence_score"), row.get("confluence"), confluence_state.get("confluence_score"), final_decision.get("confluence"), conf.get("final_confluence"))
    prob_confluence = _pick(row.get("prob_confluence"), conf.get("prob_confluence"))
    prob_meta = _pick(row.get("prob_meta"), conf.get("prob_meta"))
    hazard = _pick(row.get("hazard_score"), row.get("hazard"), snapshot.get("hazard"))
    evr_value = _pick(row.get("evr"), snapshot.get("evr"), final_decision.get("evr"))
    threshold = _pick(confluence_state.get("threshold"), final_decision.get("confluence_threshold"))
    passed = bool(_pick(confluence_state.get("passed"), snapshot.get("gates", {}).get("passed"), final_decision.get("passed"), False))
    session_bucket = _pick(confluence_state.get("session_bucket"), row.get("session_bucket"), "unknown")
    route_requested = str(state.get("routing_mode_requested") or "tree")
    route_effective = str(state.get("routing_mode_effective") or state.get("effective_routing_mode") or route_requested)
    challenger_mode = str(state.get("challenger_mode") or "tcn")
    route_note = str(state.get("routing_note") or "")

    page_header(
        "Confluence Studio",
        "Dedicated desk for rule confluence, ML confluence, specialist agreement, and routing visibility.",
        kicker="Execution coherence",
    )
    st.markdown(
        f"{status_badge(f'Route {route_effective}', 'good' if route_effective == route_requested else 'warn')} "
        f"{status_badge('Gate PASS' if passed else 'Gate BLOCKED', 'good' if passed else 'bad')} "
        f"{status_badge(f'Models {model_version}', 'neutral')}",
        unsafe_allow_html=True,
    )
    if route_note:
        inject_page_notice(route_note)

    gap = None
    if _to_float(final_confluence) is not None and _to_float(prob_confluence) is not None:
        gap = float(final_confluence) - float(prob_confluence)

    metric_grid(
        [
            {"label": "Rule Confluence", "value": _fmt(final_confluence)},
            {"label": "ML Confluence", "value": _fmt(prob_confluence)},
            {"label": "Meta Stack", "value": _fmt(prob_meta)},
            {"label": "Hazard", "value": _fmt(hazard)},
            {"label": "EVR", "value": _fmt(evr_value)},
            {"label": "Rule vs ML Gap", "value": _fmt_delta(gap)},
        ]
    )

    top_col, side_col = st.columns([1.8, 1.0])
    with top_col:
        section_title("Execution Canvas", "Current market pane under the active dashboard mode")
        render_tv_chart(market if isinstance(market, pd.DataFrame) else pd.DataFrame(), key="confluence_studio_chart")
    with side_col:
        section_title("Decision Stack", "Current execution state and route selection")
        st.json(
            {
                "route_requested": route_requested,
                "route_effective": route_effective,
                "challenger_mode": challenger_mode,
                "gate_passed": passed,
                "threshold": threshold,
                "session_bucket": session_bucket,
                "rule_confluence": final_confluence,
                "ml_confluence": prob_confluence,
                "meta_stack": prob_meta,
                "hazard": hazard,
                "evr": evr_value,
            }
        )

    tabs = st.tabs(["Confluence Reel", "Routing Matrix", "Decision Forensics"])

    with tabs[0]:
        section_title("Confluence Reel", "How rule confluence, stack confluence, and hazard evolve through recent events")
        if history.empty:
            st.info("No forward/live confluence history is available yet.")
        else:
            layer_df = history.melt(
                id_vars=["timestamp", "event_type", "trade_id", "tier", "hazard", "evr", "median_r"],
                value_vars=["final_confluence", "prob_confluence", "prob_meta"],
                var_name="signal",
                value_name="value",
            ).dropna(subset=["value"])
            line = (
                alt.Chart(layer_df)
                .mark_line(point=True, strokeWidth=2.0)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="Probability / Score"),
                    color=alt.Color(
                        "signal:N",
                        scale=alt.Scale(
                            domain=["final_confluence", "prob_confluence", "prob_meta"],
                            range=["#ffb000", "#6ea8fe", "#3ddc97"],
                        ),
                    ),
                    tooltip=["timestamp:T", "event_type:N", "trade_id:N", "signal:N", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("hazard:Q", format=".3f"), alt.Tooltip("evr:Q", format=".3f")],
                )
                .properties(height=340)
            )
            hazard_area = (
                alt.Chart(history)
                .mark_area(opacity=0.12, color="#ff6b6b")
                .encode(
                    x="timestamp:T",
                    y=alt.Y("hazard:Q", scale=alt.Scale(domain=[0, 1]), title="Hazard"),
                    tooltip=["timestamp:T", alt.Tooltip("hazard:Q", format=".3f")],
                )
            )
            st.altair_chart((hazard_area + line).interactive(), use_container_width=True)

        specialist_df = _specialist_surface(row, reasoning)
        if not specialist_df.empty:
            section_title("Specialist Ribbon", "Probability surface feeding the stack and the rule engine")
            ribbon = (
                alt.Chart(specialist_df)
                .mark_bar(cornerRadiusEnd=8, size=22)
                .encode(
                    y=alt.Y("channel:N", sort="-x", title="Channel"),
                    x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                    color=alt.value("#6ea8fe"),
                    tooltip=["channel:N", alt.Tooltip("value:Q", format=".3f")],
                )
                .properties(height=300)
            )
            st.altair_chart(ribbon, use_container_width=True)

        if not candidates.empty:
            section_title("Coherence Map", "High confluence, healthy EVR, and low hazard should cluster in the same region")
            for col in ("confluence", "evr", "hazard", "median_r", "signal_score"):
                if col in candidates.columns:
                    candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
            scatter = (
                alt.Chart(candidates.dropna(subset=["confluence", "evr"]))
                .mark_circle(opacity=0.8)
                .encode(
                    x=alt.X("confluence:Q", title="Confluence"),
                    y=alt.Y("evr:Q", title="EVR"),
                    size=alt.Size("signal_score:Q", legend=None),
                    color=alt.Color("hazard:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
                    tooltip=["timestamp", "asset", "tier", alt.Tooltip("confluence:Q", format=".3f"), alt.Tooltip("evr:Q", format=".3f"), alt.Tooltip("hazard:Q", format=".3f"), alt.Tooltip("signal_score:Q", format=".3f")],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)

    with tabs[1]:
        route_df, stack_df = _routing_tables(state)
        section_title("Routing Matrix", "Which models are currently providing specialist and stack outputs")
        left, right = st.columns(2)
        with left:
            if route_df.empty:
                st.info("No specialist route map is available in the current runtime state.")
            else:
                st.dataframe(route_df, use_container_width=True, hide_index=True)
        with right:
            if stack_df.empty:
                st.info("No stack route map is available in the current runtime state.")
            else:
                st.dataframe(stack_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        section_title("Decision Forensics", "Raw reasoning and the latest recent events feeding the confluence read")
        left, right = st.columns([1.1, 0.9])
        with left:
            if reasoning:
                st.json(reasoning)
            else:
                st.info("No reasoning payload is available for the active mode.")
        with right:
            if history.empty:
                st.info("No event history is available yet.")
            else:
                show_cols = [c for c in ["timestamp", "event_type", "trade_id", "tier", "final_confluence", "prob_confluence", "prob_meta", "hazard", "evr"] if c in history.columns]
                st.dataframe(history[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
