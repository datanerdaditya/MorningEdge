"""Dashboard views — per-page rendering logic.

Each ``render_*`` function takes the parameters it needs and renders
into the Streamlit page directly. Keeps app.py lean.
"""

from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from morningedge.dashboard.queries import (
    articles_for_class,
    asset_class_summary,
    event_breakdown_for_class,
    global_top_entities,
    latest_narratives,
    sentiment_timeline_for_class,
    top_entities_for_class,
)
from morningedge.dashboard.styling import (
    ACCENT_GOLD,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    NEGATIVE,
    POSITIVE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    sentiment_color,
)
from morningedge.taxonomy import AssetClass, by_id


# ---------------------------------------------------------------------------
# Asset class detail page
# ---------------------------------------------------------------------------


def render_asset_class_detail(asset_class_id: str, days_back: int = 2) -> None:
    """Full detail view for one asset class."""
    ac = by_id(asset_class_id)
    if ac is None:
        st.error(f"Unknown asset class: {asset_class_id}")
        return

    summary = asset_class_summary(asset_class_id, days_back=days_back)

    # --- Header ---
    st.markdown(f"# {ac.label}")
    st.markdown(
        f"<p class='me-tagline'>{_html.escape(ac.description)}</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment", f"{summary['avg_sentiment']:+.2f}")
    c2.metric("Articles", summary["n_articles"])
    c3.metric("Tier", ac.tier.upper())

    if summary["n_articles"] == 0:
        st.info(f"No articles in {ac.label} for the last {days_back} day(s).")
        return

    # --- Sentiment timeline ---
    st.markdown("## Sentiment Timeline")
    timeline = sentiment_timeline_for_class(asset_class_id, days_back=max(days_back, 7))
    if timeline.empty:
        st.caption("Not enough data yet for a timeline.")
    else:
        _render_timeline_chart(timeline)

    # --- Narratives ---
    st.markdown("## Active Narratives")
    narratives = latest_narratives(asset_class_id=asset_class_id, limit=10)
    if narratives.empty:
        st.caption("No narratives have formed in this class yet.")
    else:
        for _, n in narratives.iterrows():
            _render_narrative_card(n)

    # --- Articles ---
    st.markdown("## Articles")
    articles = articles_for_class(asset_class_id, limit=30, days_back=days_back)
    if articles.empty:
        st.caption("No articles.")
    else:
        _render_articles_table(articles)

    # --- Entity leaderboard ---
    st.markdown("## Top Entities Mentioned")
    entities = top_entities_for_class(
        asset_class_id,
        days_back=days_back,
        label_filter=["company", "person", "ticker"],
        limit=15,
    )
    if entities.empty:
        st.caption("No entities yet.")
    else:
        _render_entity_table(entities)

    # --- Event distribution ---
    st.markdown("## Event Distribution")
    events = event_breakdown_for_class(asset_class_id, days_back=days_back)
    if events.empty:
        st.caption("No events classified yet.")
    else:
        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Global entities page
# ---------------------------------------------------------------------------


def render_entities_page(days_back: int = 2) -> None:
    """Global view: most-mentioned entities across all classes."""
    st.markdown("# Entities")
    st.markdown(
        "<p class='me-tagline'>Most-mentioned companies, tickers, and people across the lookback window.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Companies", "People", "Tickers"])

    with tabs[0]:
        df = global_top_entities(days_back=days_back, label_filter=["company"], limit=30)
        _render_entity_table(df)

    with tabs[1]:
        df = global_top_entities(days_back=days_back, label_filter=["person"], limit=20)
        _render_entity_table(df)

    with tabs[2]:
        df = global_top_entities(days_back=days_back, label_filter=["ticker"], limit=20)
        _render_entity_table(df)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------


def _render_timeline_chart(df: pd.DataFrame) -> None:
    """A clean line chart of daily avg sentiment for one class."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["avg_sentiment"],
            mode="lines+markers",
            line=dict(color=ACCENT_GOLD, width=2),
            marker=dict(size=8, color=ACCENT_GOLD),
            hovertemplate=(
                "<b>%{x|%b %d}</b><br>"
                "Sentiment: %{y:+.2f}<br>"
                "Articles: %{customdata}<extra></extra>"
            ),
            customdata=df["n_articles"],
        )
    )
    # Zero line for reference
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED, opacity=0.4)
    fig.update_layout(
        plot_bgcolor=BG_PRIMARY,
        paper_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT_SECONDARY, family="Inter"),
        xaxis=dict(showgrid=False, tickformat="%b %d"),
        yaxis=dict(showgrid=True, gridcolor=BG_TERTIARY, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_narrative_card(n: pd.Series) -> None:
    """One narrative card — same style as the home page."""
    meta = (
        f"{_html.escape(str(n['asset_class']))} &middot; "
        f"{n['article_count']} articles &middot; {n['narrative_date']}"
    )
    title = _html.escape(str(n["title"]))
    summary_text = _html.escape(str(n["summary"]))
    st.markdown(
        "<div class='me-card'>"
        f"<div class='me-card-meta'>{meta}</div>"
        f"<div class='me-card-title'>{title}</div>"
        f"<div style='color:#9AA0A8; font-size:0.95rem;'>{summary_text}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_articles_table(df: pd.DataFrame) -> None:
    """Compact article table with sentiment-coloured score column."""
    # Prepare a display-friendly copy
    display_df = df[
        ["title", "source_id", "sentiment_score", "sentiment_label", "event_type"]
    ].copy()
    display_df["sentiment_score"] = display_df["sentiment_score"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
    )
    display_df.columns = ["Title", "Source", "Score", "Label", "Event"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn(width="large"),
            "Source": st.column_config.TextColumn(width="medium"),
        },
    )


def _render_entity_table(df: pd.DataFrame) -> None:
    """Entity leaderboard: name, mentions, avg sentiment."""
    if df.empty:
        st.caption("No entities for the selected filter.")
        return

    display_df = df[["entity", "label", "mentions", "avg_sentiment"]].copy()
    display_df["avg_sentiment"] = display_df["avg_sentiment"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
    )
    display_df.columns = ["Entity", "Type", "Mentions", "Avg Sentiment"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)