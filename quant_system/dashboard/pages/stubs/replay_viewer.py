from __future__ import annotations

from quant_system.dashboard.data_access import build_context
from quant_system.dashboard.pages.apps.replay_timeline import render_replay_timeline


def render_replay_viewer(theme_choice: str = "bloomberg") -> None:
    context = build_context(theme_choice)
    render_replay_timeline(theme_choice, context.model_version, context=context)


if __name__ == "__main__":
    render_replay_viewer()
