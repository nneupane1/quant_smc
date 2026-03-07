"""Standalone Streamlit wrapper for the canonical trade-log dashboard page."""

from __future__ import annotations

import streamlit as st

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.pages.apps.trade_log import render_trade_log
from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter
from quant_system.utils.logger import runtime_logged


@runtime_logged("Trade log page runtime")
def main() -> None:
    st.set_page_config(page_title="Trade Log", layout="wide")
    if "dashboard_adapter" not in st.session_state:
        st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()

    context = build_context("Bloomberg", adapter=st.session_state["dashboard_adapter"])
    render_trade_log("Bloomberg", context.model_version, context=context)


if __name__ == "__main__":
    main()
