"""MorningEdge — dashboard entry point.

Run with:
    streamlit run src/morningedge/dashboard/app.py
"""

from __future__ import annotations

import html as _html
import sys
from pathlib import Path

# Allow running directly via `streamlit run src/morningedge/dashboard/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from morningedge.dashboard.queries import (
    all_asset_class_summaries,
    latest_narratives,
    overall_summary,
    regime_label,
    top_narrative_for_class,
)
from morningedge.dashboard.styling import CSS, sentiment_color
from morningedge.dashboard.views import (
    render_asset_class_detail,
    render_entities_page,
)
from morningedge.taxonomy import TAXONOMY, by_tier


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MorningEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def cached_overall(days_back: int = 1) -> dict:
    return overall_summary(days_back=days_back)


@st.cache_data(ttl=300)
def cached_classes(days_back: int = 1) -> list[dict]:
    return all_asset_class_summaries(days_back=days_back)


@st.cache_data(ttl=300)
def cached_top_narratives(asset_class_id: str) -> dict | None:
    return top_narrative_for_class(asset_class_id)


@st.cache_data(ttl=300)
def cached_recent_narratives(limit: int = 10):
    return latest_narratives(limit=limit)


# ---------------------------------------------------------------------------
# Card renderer (shared on the Overview page)
# ---------------------------------------------------------------------------


def _render_class_card(summary: dict) -> None:
    """One asset class tile."""
    color = sentiment_color(summary["avg_sentiment"])
    narrative = cached_top_narratives(summary["asset_class_id"])

    label = _html.escape(summary["label"])
    tier = _html.escape(summary["tier"].upper())
    n = summary["n_articles"]
    score = f"{summary['avg_sentiment']:+.2f}"

    parts = [
        "<div class='me-card'>",
        f"<div class='me-card-meta'>{tier} &middot; {n} articles</div>",
        f"<div class='me-card-title'>{label}</div>",
        (
            f"<div style='font-family:\"JetBrains Mono\",monospace; "
            f"font-size:1.5rem; color:{color}; margin:0.4rem 0;'>{score}</div>"
        ),
    ]

    if narrative is not None:
        title = _html.escape(str(narrative["title"]))
        summary_text = _html.escape(str(narrative["summary"]))
        parts.append(
            f"<div style='font-family:\"Cormorant Garamond\",serif; "
            f"font-size:1.05rem; color:#E8E6E1; margin-top:0.6rem; "
            f"font-style:italic;'>{title}</div>"
        )
        parts.append(
            f"<div style='font-size:0.85rem; color:#9AA0A8; "
            f"margin-top:0.2rem;'>{summary_text}</div>"
        )
    else:
        parts.append(
            "<div style='color:#5A6068; font-size:0.85rem; "
            "margin-top:0.6rem;'>No active narrative</div>"
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — global controls + page routing
# ---------------------------------------------------------------------------


with st.sidebar:
    st.markdown("### Navigate")

    page = st.radio(
        "View",
        ["Overview", "Asset Class Detail", "Entities", "Ask MorningEdge"],
        label_visibility="collapsed",
    )


    selected_class_id = None
    if page == "Asset Class Detail":
        labels = [f"{ac.label}  ·  {ac.tier}" for ac in TAXONOMY]
        chosen = st.selectbox("Choose an asset class", labels)
        idx = labels.index(chosen)
        selected_class_id = TAXONOMY[idx].id

    st.markdown("---")
    st.markdown("### Settings")
    days_back = st.slider("Lookback window (days)", 1, 7, 2)
    st.caption(f"Showing data from the last {days_back} day(s).")
    st.markdown("---")
    st.caption("MorningEdge v0.1 · Built by Aditya")


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------


if page == "Asset Class Detail" and selected_class_id:
    render_asset_class_detail(selected_class_id, days_back=days_back)

elif page == "Entities":
    render_entities_page(days_back=days_back)

elif page == "Ask MorningEdge":
    from morningedge.dashboard.views import render_chat_page
    render_chat_page(lookback_days=days_back)

else:
    # ----------------------- OVERVIEW -----------------------
    st.markdown("# MorningEdge")
    st.markdown(
        "<p class='me-tagline'>An AI-powered morning brief for the leveraged loans and private credit market.</p>",
        unsafe_allow_html=True,
    )

    overall = cached_overall(days_back=days_back)
    regime = regime_label(overall["avg_sentiment"])

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.metric("Regime", regime)
    c2.metric("Avg Sentiment", f"{overall['avg_sentiment']:+.2f}")
    c3.metric("Articles", overall["n_articles"])
    c4.metric("Positive / Negative", f"{overall['n_positive']} / {overall['n_negative']}")

    summaries = {s["asset_class_id"]: s for s in cached_classes(days_back=days_back)}

    # Hero tier
    st.markdown("## Hero — Leveraged Finance & Private Credit")
    hero_cols = st.columns(4)
    for col, ac in zip(hero_cols, by_tier("hero")):
        with col:
            _render_class_card(summaries[ac.id])

    # Macro tier
    st.markdown("## Macro Context")
    macro_cols = st.columns(3)
    for col, ac in zip(macro_cols, by_tier("macro")):
        with col:
            _render_class_card(summaries[ac.id])

    # Breadth tier (collapsible)
    with st.expander("Breadth — Sectors, Regions, Commodities, FX", expanded=False):
        breadth = by_tier("breadth")
        for i in range(0, len(breadth), 4):
            row = breadth[i : i + 4]
            cols = st.columns(4)
            for col, ac in zip(cols, row):
                with col:
                    _render_class_card(summaries[ac.id])

    # Recent narratives
    st.markdown("## Recent Narratives")
    narr_df = cached_recent_narratives(limit=15)

    if narr_df.empty:
        st.info("No narratives yet. Run `python scripts/cluster_narratives.py`.")
    else:
        for _, n in narr_df.iterrows():
            meta = (
                f"{_html.escape(str(n['asset_class']))} &middot; "
                f"{n['article_count']} articles &middot; {n['narrative_date']}"
            )
            title = _html.escape(str(n["title"]))