from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def _open_trades_frame(open_trades) -> pd.DataFrame:
    rows = []
    if isinstance(open_trades, dict):
        iterable = open_trades.items()
    elif isinstance(open_trades, list):
        iterable = enumerate(open_trades)
    else:
        iterable = []

    for trade_id, trade in iterable:
        if isinstance(trade, dict):
            rows.append(
                {
                    "trade_id": trade.get("trade_id", trade_id),
                    "asset": trade.get("asset"),
                    "leg": trade.get("leg") or trade.get("metadata", {}).get("leg") if isinstance(trade.get("metadata"), dict) else trade.get("leg"),
                    "side": trade.get("side"),
                    "entry_price": trade.get("entry_price") or trade.get("entry"),
                    "stop_price": trade.get("stop_price") or trade.get("stop"),
                    "size_usd": trade.get("size_usd") or trade.get("size"),
                    "highest_r": trade.get("highest_r"),
                }
            )
        else:
            rows.append(
                {
                    "trade_id": getattr(trade, "trade_id", trade_id),
                    "asset": getattr(trade, "asset", None),
                    "leg": getattr(trade, "metadata", {}).get("leg") if hasattr(trade, "metadata") and isinstance(getattr(trade, "metadata"), dict) else getattr(trade, "leg", None),
                    "side": getattr(trade, "side", None),
                    "entry_price": getattr(trade, "entry_price", None),
                    "stop_price": getattr(trade, "stop_price", None),
                    "size_usd": getattr(trade, "size_usd", None),
                    "highest_r": getattr(trade, "highest_r", None),
                }
            )
    return pd.DataFrame(rows)


def render_forward_test(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    forward = context.forward
    state = forward["state"]
    events = forward["events"]
    candles = forward["candles"] if isinstance(forward["candles"], pd.DataFrame) else pd.DataFrame()

    page_header(
        "Forward Test Cockpit",
        "Operator view for the live paper engine: capital cycle, flow model, confluence, hazard, and reasoning stream.",
        kicker="Forward Loop",
    )
    cooling_to = state.get("cooling_to")
    st.markdown(
        f"{status_badge('Compounding active', 'good')} "
        f"{status_badge(f'Cooling until {cooling_to}' if cooling_to else 'Cooling off', 'warn' if cooling_to else 'neutral')}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Equity", "value": f"${state.get('equity', 0.0):,.2f}"},
            {"label": "Locked Profit", "value": f"${state.get('locked_profit', 0.0):,.2f}"},
            {"label": "Free Capital", "value": f"${state.get('free_capital', 0.0):,.2f}"},
            {"label": "1h Flow ML", "value": f"{state.get('flow_1h', 0.0):.3f}" if state.get("flow_1h") is not None else "--"},
            {"label": "Confluence", "value": f"{state.get('confluence', 0.0):.3f}" if state.get("confluence") is not None else "--"},
            {"label": "Hazard", "value": f"{state.get('hazard', 0.0):.3f}" if state.get("hazard") is not None else "--"},
        ]
    )

    left, right = st.columns([2.1, 1.1])
    with left:
        section_title("Market View", "Current candle feed for the active forward adapter")
        render_tv_chart(candles, key="forward_candles")
        section_title("Execution Tape", "Recent engine events")
        if events:
            st.dataframe(pd.DataFrame(events[-100:]), use_container_width=True, hide_index=True)
        else:
            st.info("No forward events yet.")

    with right:
        section_title("Open Trades", "Core and runner legs")
        open_trades = _open_trades_frame(state.get("open_trades", {}))
        if open_trades.empty:
            st.info("No open trades.")
        else:
            st.dataframe(open_trades, use_container_width=True, hide_index=True)

        section_title("Latest Reasoning", "Most recent high-signal event payload")
        latest = events[-1] if events else {}
        payload = latest.get("payload", {}) if isinstance(latest, dict) else {}
        reasoning = payload.get("reasoning", payload)
        if reasoning:
            st.json(reasoning)
        else:
            st.info("No reasoning payload captured yet.")
