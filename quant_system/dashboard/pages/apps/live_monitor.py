from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_system.dashboard.components.js.tv_chart.tv_chart import render_tv_chart
from quant_system.dashboard.data_access import DashboardContext
from quant_system.dashboard.ui import metric_grid, page_header, section_title, status_badge


def render_live(theme_choice: str, model_version: str, *, context: DashboardContext) -> None:
    forward = context.forward
    state = forward["state"]
    events = forward["events"]
    candles = forward["candles"] if isinstance(forward["candles"], pd.DataFrame) else pd.DataFrame()
    live_enabled = bool(st.session_state.get("live_trading_enabled", False))

    page_header(
        "Live Trading Cockpit",
        "Bloomberg-style operator shell for paper/live supervision. Streamlit remains the control plane, with custom component hooks still available.",
        kicker="Live Monitor",
    )
    st.markdown(
        f"{status_badge('Live routing enabled' if live_enabled else 'Paper routing only', 'good' if live_enabled else 'warn')} "
        f"{status_badge(f'Models {model_version}', 'neutral')}",
        unsafe_allow_html=True,
    )

    metric_grid(
        [
            {"label": "Equity", "value": f"${state.get('equity', 0.0):,.2f}"},
            {"label": "Locked", "value": f"${state.get('locked_profit', 0.0):,.2f}"},
            {"label": "Free", "value": f"${state.get('free_capital', 0.0):,.2f}"},
            {"label": "Hazard", "value": f"{state.get('hazard', 0.0):.3f}" if state.get("hazard") is not None else "--"},
        ]
    )

    chart_col, panel_col = st.columns([2.2, 1.0])
    with chart_col:
        section_title("Execution Chart", "Current market feed and overlay surface")
        render_tv_chart(candles, key="live_chart")
    with panel_col:
        section_title("Strategy State", "Capital cycle and gating surface")
        st.json(
            {
                "confluence": state.get("confluence"),
                "evr": state.get("evr"),
                "flow_1h": state.get("flow_1h"),
                "hazard": state.get("hazard"),
                "cooling_to": state.get("cooling_to"),
                "open_trades": len(state.get("open_trades", {})),
            }
        )

    section_title("Event Stream", "Latest orchestration events")
    if events:
        st.dataframe(pd.DataFrame(events[-100:]), use_container_width=True, hide_index=True)
    else:
        st.info("No live events are available yet.")
