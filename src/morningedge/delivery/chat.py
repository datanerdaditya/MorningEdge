"""RAG-powered chat over MorningEdge data.

Pipeline per question:
    1. Embed the question with mpnet.
    2. Retrieve the top-K most-similar recent articles from the DB.
    3. Format them as numbered sources.
    4. Prompt Gemini Pro with system instructions + sources + question.
    5. Stream the response back to the caller.

The whole thing is grounded: Gemini cannot cite an article we didn't retrieve,
and we tell it not to use outside knowledge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
from loguru import logger

from morningedge.ingestion.dedup import embed_texts
from morningedge.llm.gemini import MODEL_FLASH, _get_client, _wait_for_quota, _bump_quota
from morningedge.llm.prompts import RAG_SYSTEM, RAG_USER_TEMPLATE
from morningedge.storage.db import connect


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = 8                # how many articles to retrieve per question
DEFAULT_LOOKBACK_DAYS = 7


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedArticle:
    """One article surfaced by retrieval, ready to format into a prompt."""

    article_id: str
    title: str
    description: str | None
    source_id: str
    canonical_url: str
    published_at: datetime
    sentiment_score: float | None
    similarity: float


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    question: str,
    top_k: int = TOP_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[RetrievedArticle]:
    """Embed the question and pull the K most-similar recent articles."""
    if not question.strip():
        return []

    q_vec = embed_texts([question])[0]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT article_id, title, description, source_id, canonical_url,
                   published_at, sentiment_score, embedding
            FROM articles
            WHERE embedding IS NOT NULL
              AND published_at >= ?
            """,
            [cutoff],
        ).fetchall()

    if not rows:
        return []

    candidates = []
    for row in rows:
        aid, title, desc, source, url, pub, sent, emb_json = row
        if emb_json is None:
            continue
        try:
            # DuckDB returns the JSON column as a string; parse it first.
            if isinstance(emb_json, str):
                emb_data = json.loads(emb_json)
            else:
                emb_data = emb_json
            vec = np.asarray(emb_data, dtype=np.float32)
            sim = float(vec @ q_vec)
            candidates.append((sim, aid, title, desc, source, url, pub, sent))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    candidates.sort(reverse=True)
    top = candidates[:top_k]

    return [
        RetrievedArticle(
            article_id=aid,
            title=title,
            description=desc,
            source_id=source,
            canonical_url=url,
            published_at=pub,
            sentiment_score=float(sent) if sent is not None else None,
            similarity=sim,
        )
        for sim, aid, title, desc, source, url, pub, sent in top
    ]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_sources(articles: list[RetrievedArticle]) -> str:
    """Render retrieved articles as numbered sources for the prompt."""
    lines = []
    for i, a in enumerate(articles, start=1):
        date_str = a.published_at.strftime("%Y-%m-%d") if a.published_at else "n/a"
        sent_str = f" sentiment={a.sentiment_score:+.2f}" if a.sentiment_score is not None else ""
        body = f"[{i}] ({date_str}, {a.source_id}{sent_str}) {a.title}"
        if a.description:
            body += f" — {a.description[:300]}"
        lines.append(body)
    return "\n\n".join(lines)


def build_prompt(question: str, articles: list[RetrievedArticle]) -> str:
    """Assemble the full system + user prompt."""
    sources = _format_sources(articles) if articles else "(no relevant sources found)"
    return RAG_SYSTEM + "\n\n" + RAG_USER_TEMPLATE.format(
        sources=sources,
        question=question.strip(),
    )


# ---------------------------------------------------------------------------
# Public chat API
# ---------------------------------------------------------------------------


def answer(
    question: str,
    top_k: int = TOP_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[str, list[RetrievedArticle]]:
    """Synchronous: return (answer_text, sources). Useful for testing."""
    articles = retrieve(question, top_k=top_k, lookback_days=lookback_days)

    if not articles:
        return (
            "I don't have any indexed articles relevant to that question yet. "
            "Try running the pipeline (`python scripts/run_pipeline.py`) to ingest fresh data.",
            [],
        )

    prompt = build_prompt(question, articles)
    client = _get_client()
    _wait_for_quota(MODEL_FLASH)

    response = client.models.generate_content(
        model=MODEL_FLASH,
        contents=prompt,
    )
    _bump_quota(MODEL_FLASH)
    return (response.text or "", articles)


def answer_stream(
    question: str,
    top_k: int = TOP_K,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[Iterable[str], list[RetrievedArticle]]:
    """Streaming: return (token_iterator, sources).

    Streams chunks as they arrive from Gemini. The dashboard uses this so
    answers appear progressively rather than as a wall of text.
    """
    articles = retrieve(question, top_k=top_k, lookback_days=lookback_days)

    if not articles:
        def _empty() -> Iterable[str]:
            yield (
                "I don't have any indexed articles relevant to that question yet. "
                "Try running the pipeline first."
            )
        return (_empty(), [])

    prompt = build_prompt(question, articles)
    client = _get_client()
    _wait_for_quota(MODEL_FLASH)

    def _stream() -> Iterable[str]:
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL_FLASH,
                contents=prompt,
            ):
                if chunk.text:
                    yield chunk.text
            _bump_quota(MODEL_FLASH)
        except Exception as e:
            logger.warning(f"Gemini stream failed: {e}")
            yield f"\n\n_(answer interrupted: {type(e).__name__})_"

    return (_stream(), articles)