"""Smoke tests for the ingestion data models."""

from datetime import UTC, datetime

from morningedge.ingestion.models import Article, SourceTier, make_article_id


def test_make_article_id_deterministic():
    """Same URL must always produce the same id."""
    id1 = make_article_id("https://example.com/article")
    id2 = make_article_id("https://example.com/article")
    assert id1 == id2
    assert len(id1) == 16


def test_make_article_id_distinct():
    """Different URLs must produce different ids."""
    id1 = make_article_id("https://example.com/a")
    id2 = make_article_id("https://example.com/b")
    assert id1 != id2


def test_article_normalises_title_whitespace():
    a = Article(
        article_id=make_article_id("https://x.com/1"),
        title="   Fed holds rates steady   ",
        url="https://x.com/1",
        canonical_url="https://x.com/1",
        source_id="reuters_business",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
    )
    assert a.title == "Fed holds rates steady"


def test_article_forces_utc():
    """Naive datetime should be coerced to UTC, not rejected."""
    a = Article(
        article_id=make_article_id("https://x.com/1"),
        title="Test",
        url="https://x.com/1",
        canonical_url="https://x.com/1",
        source_id="reuters_business",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 1, 12, 0),  # naive
    )
    assert a.published_at.tzinfo is not None