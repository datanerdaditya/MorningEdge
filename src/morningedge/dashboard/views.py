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
    BG_TERTIARY,
    TEXT_MUTED,
    TEXT_SECONDARY,
)
from morningedge.taxonomy import by_id

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

    # ---------------------------------------------------------------------------
# RAG chat page (Day 12)
# ---------------------------------------------------------------------------


def render_chat_page(lookback_days: int = 7) -> None:
    """Conversational interface over the MorningEdge corpus."""
    import html as _html_mod

    from morningedge.delivery.chat import answer_stream

    st.markdown("# Ask MorningEdge")
    st.markdown(
        "<p class='me-tagline'>"
        "Ask anything about the credit news in your database. Answers are grounded "
        "in real articles — every claim is cited."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Sample questions to seed the user ---
    samples = [
        "What's the latest on private credit?",
        "Why is rates sentiment positive today?",
        "Are there any defaults or distressed stories?",
        "What's happening with the Fed and ECB?",
        "Any major M&A activity this week?",
    ]
    st.caption("Try one of these:")
    sample_cols = st.columns(len(samples))
    for col, q in zip(sample_cols, samples, strict=False):
        if col.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_question = q

    # --- Initialise chat history ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role", "content", "sources"}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # --- Render existing history ---
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"Sources ({len(turn['sources'])})"):
                    for i, s in enumerate(turn["sources"], start=1):
                        date_str = s["published_at"].strftime("%Y-%m-%d") if s["published_at"] else "n/a"
                        st.markdown(
                            f"**[{i}]** [{_html_mod.escape(s['title'])}]({s['canonical_url']})  \n"
                            f"_{s['source_id']} · {date_str} · "
                            f"sim {s['similarity']:.2f}_"
                        )

    # --- Input box (or pending sample question) ---
    user_q = st.chat_input("Ask a question...")
    if st.session_state.pending_question and not user_q:
        user_q = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_q:
        # Append user turn
        st.session_state.chat_history.append(
            {"role": "user", "content": user_q, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(user_q)

        # Stream the assistant's answer
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Searching the corpus..."):
                stream, articles = answer_stream(user_q, lookback_days=lookback_days)

            full_text = ""
            for chunk in stream:
                full_text += chunk
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)

            # Show sources inline
            if articles:
                with st.expander(f"Sources ({len(articles)})"):
                    for i, a in enumerate(articles, start=1):
                        date_str = a.published_at.strftime("%Y-%m-%d") if a.published_at else "n/a"
                        st.markdown(
                            f"**[{i}]** [{_html_mod.escape(a.title)}]({a.canonical_url})  \n"
                            f"_{a.source_id} · {date_str} · sim {a.similarity:.2f}_"
                        )

        # Persist into history (with serialisable sources)
        sources_serialised = [
            {
                "title": a.title,
                "canonical_url": a.canonical_url,
                "source_id": a.source_id,
                "published_at": a.published_at,
                "similarity": a.similarity,
            }
            for a in articles
        ]
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_text, "sources": sources_serialised}
        )

        # ---------------------------------------------------------------------------
# Daily brief page (Day 13)
# ---------------------------------------------------------------------------


def render_brief_page() -> None:
    """Today's morning brief, with a button to regenerate."""

    from morningedge.delivery.brief import generate_brief, get_latest_brief

    st.markdown("# Daily Brief")
    st.markdown(
        "<p class='me-tagline'>"
        "A 90-second morning read. Written by Gemini Pro from MorningEdge's enriched data."
        "</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        regenerate = st.button("Generate brief", type="primary")
    with col2:
        st.caption("Generates a new brief from today's data. Costs one Gemini Pro call.")

    if regenerate:
        with st.spinner("Drafting brief... (one Gemini Pro call)"):
            try:
                brief = generate_brief(days_back=2)
                st.success("Brief generated.")
            except Exception as e:
                st.error(f"Brief generation failed: {e}")
                return
    else:
        brief = get_latest_brief()

    if brief is None:
        st.info(
            "No brief yet. Click **Generate brief** above, or run "
            "`python scripts/generate_brief.py` from the terminal."
        )
        return

    # --- Brief metadata strip ---
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Date", str(brief.brief_date))
    cm2.metric("Regime", brief.regime)
    cm3.metric("Articles synthesised", brief.n_articles)

    st.markdown("---")

    # --- The brief itself ---
    st.markdown(brief.body_markdown)