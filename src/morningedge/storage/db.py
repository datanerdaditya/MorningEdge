"""DuckDB connection management and CRUD operations.

All database access in MorningEdge goes through this module. No other
module should import duckdb directly — that keeps SQL out of the
business logic and makes future migrations painless.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb
from loguru import logger

from morningedge.config import settings
from morningedge.ingestion.models import Article
from morningedge.storage.schema import ALL_DDL


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection, ensuring it's closed afterwards.

    Usage:
        with connect() as conn:
            conn.execute("SELECT COUNT(*) FROM articles").fetchone()
    """
    path = db_path or settings.duckdb_path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(path))
    try:
        yield conn
    finally:
        conn.close()


def init_schema(db_path: Path | None = None) -> None:
    """Create all tables and indexes if they don't exist.

    Idempotent: safe to call on every pipeline run.
    """
    with connect(db_path) as conn:
        for ddl in ALL_DDL:
            conn.execute(ddl)
    logger.info(f"Schema initialised at {db_path or settings.duckdb_path}")


# ---------------------------------------------------------------------------
# Article persistence
# ---------------------------------------------------------------------------


def insert_articles(
    articles: list[Article],
    db_path: Path | None = None,
) -> tuple[int, int]:
    """Insert articles, skipping duplicates by article_id.

    Returns
    -------
    (inserted, skipped)
        ``inserted`` is the number of new rows added.
        ``skipped`` is the number that already existed (idempotent re-runs).
    """
    if not articles:
        return (0, 0)

    with connect(db_path) as conn:
        # Get existing ids in one query (faster than checking each)
        ids = [a.article_id for a in articles]
        existing = {
            row[0]
            for row in conn.execute(
                f"SELECT article_id FROM articles WHERE article_id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
        }

        new_articles = [a for a in articles if a.article_id not in existing]
        if not new_articles:
            return (0, len(articles))

        rows = [_article_to_row(a) for a in new_articles]
        conn.executemany(
            """
            INSERT INTO articles (
                article_id, title, url, canonical_url,
                source_id, source_tier, description,
                published_at, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    inserted = len(new_articles)
    skipped = len(articles) - inserted
    logger.info(f"DB: inserted {inserted}, skipped {skipped} (already present)")
    return (inserted, skipped)


def count_articles(db_path: Path | None = None) -> int:
    """Total articles currently in the database."""
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def recent_articles(
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict]:
    """Return the N most recently fetched articles as a list of dicts."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT article_id, title, source_id, source_tier,
                   published_at, fetched_at, canonical_url
            FROM articles
            ORDER BY published_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        cols = [desc[0] for desc in conn.description]
        return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _article_to_row(a: Article) -> tuple:
    """Marshal an Article into the tuple shape DuckDB expects."""
    return (
        a.article_id,
        a.title,
        str(a.url),
        a.canonical_url,
        a.source_id,
        a.source_tier.value,
        a.description,
        a.published_at,
        a.fetched_at,
    )