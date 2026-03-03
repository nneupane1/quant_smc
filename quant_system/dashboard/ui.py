from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st


def page_header(title: str, subtitle: str, kicker: Optional[str] = None) -> None:
    kicker_html = f"<div class='qs-kicker'>{kicker}</div>" if kicker else ""
    st.markdown(
        f"""
        <section class="qs-hero">
            {kicker_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: Optional[str] = None) -> None:
    caption_html = f"<span>{caption}</span>" if caption else ""
    st.markdown(
        f"""
        <div class="qs-section-head">
            <h3>{title}</h3>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, tone: str = "neutral") -> str:
    tone = tone if tone in {"good", "bad", "warn", "neutral"} else "neutral"
    return f"<span class='qs-badge qs-badge-{tone}'>{label}</span>"


def metric_grid(metrics: Iterable[dict]) -> None:
    items = list(metrics)
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        tone = item.get("tone", "normal")
        delta = item.get("delta")
        delta_color = item.get("delta_color", "normal")
        with col:
            st.metric(item["label"], item["value"], delta=delta, delta_color=delta_color)
            if tone != "normal":
                st.markdown(
                    f"<div class='qs-metric-tone qs-metric-tone-{tone}'></div>",
                    unsafe_allow_html=True,
                )


def inject_page_notice(text: str) -> None:
    st.markdown(f"<div class='qs-notice'>{text}</div>", unsafe_allow_html=True)
