"""Tests for the fuzzy dedup layer.

These tests don't touch the DB or the network — they verify the
embedding + similarity logic on synthetic inputs.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from morningedge.ingestion.dedup import (
    SIMILARITY_THRESHOLD,
    embed_texts,
)
from morningedge.ingestion.models import Article, SourceTier, make_article_id


def _make_article(url: str, title: str) -> Article:
    return Article(
        article_id=make_article_id(url),
        title=title,
        url=url,
        canonical_url=url,
        source_id="ft_alphaville",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )


def test_embeddings_are_normalised():
    """Sentence-transformer with normalize=True should give unit vectors."""
    vecs = embed_texts(["Fed holds rates steady", "ECB signals September cut"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_identical_titles_have_similarity_one():
    text = "JPMorgan beats Q3 earnings on trading revenue"
    vecs = embed_texts([text, text])
    assert float(vecs[0] @ vecs[1]) > 0.999


def test_paraphrased_titles_above_threshold():
    """Two phrasings of the same news event should cluster as duplicates."""
    a = "Federal Reserve holds interest rates steady at 4.5%"
    b = "Fed keeps interest rates unchanged at 4.5 percent"
    vecs = embed_texts([a, b])
    sim = float(vecs[0] @ vecs[1])
    assert sim >= SIMILARITY_THRESHOLD, f"sim={sim:.3f}, expected >= {SIMILARITY_THRESHOLD}"


def test_near_verbatim_titles_above_threshold():
    """Same wire copy with tiny edits should be flagged as duplicates."""
    a = "JPMorgan beats Q3 earnings on trading revenue"
    b = "JPMorgan beats Q3 earnings, driven by trading revenue"
    vecs = embed_texts([a, b])
    sim = float(vecs[0] @ vecs[1])
    assert sim >= SIMILARITY_THRESHOLD


def test_different_topics_below_threshold():
    """Unrelated headlines should not collide."""
    a = "Apple announces new iPhone in September event"
    b = "ECB cuts rates by 25bps citing weak eurozone growth"
    vecs = embed_texts([a, b])
    sim = float(vecs[0] @ vecs[1])
    assert sim < 0.5, f"sim={sim:.3f}, unexpectedly high"


def test_related_but_distinct_below_threshold():
    """Same topic, distinct news events should NOT be merged."""
    a = "Fed cuts rates by 25 basis points"
    b = "Fed signals more cuts ahead in upcoming meetings"
    vecs = embed_texts([a, b])
    sim = float(vecs[0] @ vecs[1])
    assert sim < SIMILARITY_THRESHOLD, (
        f"sim={sim:.3f} — these are distinct news events, threshold may need raising"
    )


def test_different_topics_below_threshold():
    """Unrelated headlines should not collide."""
    a = "Apple announces new iPhone in September event"
    b = "ECB cuts rates by 25bps citing weak eurozone growth"
    vecs = embed_texts([a, b])
    sim = float(vecs[0] @ vecs[1])
    assert sim < 0.7, f"sim={sim:.3f}, unexpectedly high"