from __future__ import annotations

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.pages.apps.live_monitor import render_live


def render_forward_monitor(theme_choice: str = "bloomberg") -> None:
    context = build_context(theme_choice)
    render_live(theme_choice, context.model_version, context=context)


if __name__ == "__main__":
    render_forward_monitor()
