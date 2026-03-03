"""Standalone Streamlit wrapper for the canonical forward dashboard page."""

from __future__ import annotations

import streamlit as st

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.pages.apps.forward_test import render_forward_test
from quant_system.forward_test.forward_dashboard_adapter import ForwardDashboardAdapter


def main() -> None:
    st.set_page_config(page_title="Forward Test Cockpit", layout="wide")
    if "dashboard_adapter" not in st.session_state:
        st.session_state["dashboard_adapter"] = ForwardDashboardAdapter()

    context = build_context("Bloomberg", adapter=st.session_state["dashboard_adapter"])
    render_forward_test("Bloomberg", context.model_version, context=context)


if __name__ == "__main__":
    main()
