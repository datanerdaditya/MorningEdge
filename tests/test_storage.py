"""Tests for the DuckDB storage layer."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from morningedge.ingestion.models import Article, SourceTier, make_article_id
from morningedge.storage.db import (
    count_articles,
    init_schema,
    insert_articles,
    recent_articles,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh DuckDB file per test."""
    db = tmp_path / "test.duckdb"
    init_schema(db)
    return db


def _make_article(url: str, title: str = "Test headline") -> Article:
    return Article(
        article_id=make_article_id(url),
        title=title,
        url=url,
        canonical_url=url,
        source_id="ft_alphaville",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
    )


def test_init_schema_creates_empty_db(tmp_db: Path):
    assert count_articles(tmp_db) == 0


def test_insert_articles_basic(tmp_db: Path):
    articles = [_make_article(f"https://x.com/{i}") for i in range(5)]
    inserted, skipped = insert_articles(articles, tmp_db)
    assert inserted == 5
    assert skipped == 0
    assert count_articles(tmp_db) == 5


def test_insert_is_idempotent(tmp_db: Path):
    """Inserting the same articles twice should skip duplicates."""
    articles = [_make_article(f"https://x.com/{i}") for i in range(3)]

    inserted1, skipped1 = insert_articles(articles, tmp_db)
    inserted2, skipped2 = insert_articles(articles, tmp_db)

    assert inserted1 == 3 and skipped1 == 0
    assert inserted2 == 0 and skipped2 == 3
    assert count_articles(tmp_db) == 3


def test_recent_articles_returns_correct_shape(tmp_db: Path):
    articles = [_make_article(f"https://x.com/{i}", f"Title {i}") for i in range(3)]
    insert_articles(articles, tmp_db)

    rows = recent_articles(limit=10, db_path=tmp_db)
    assert len(rows) == 3
    assert "title" in rows[0]
    assert "source_id" in rows[0]
    assert rows[0]["source_id"] == "ft_alphaville"


def test_empty_insert_is_noop(tmp_db: Path):
    inserted, skipped = insert_articles([], tmp_db)
    assert inserted == 0
    assert skipped == 0
    