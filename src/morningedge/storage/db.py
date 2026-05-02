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

# ---------------------------------------------------------------------------
# Enrichment persistence (Week 2)
# ---------------------------------------------------------------------------


def get_unenriched_articles(
    limit: int = 500,
    db_path: Path | None = None,
) -> list[dict]:
    """Return articles that haven't been enriched yet (enriched_at IS NULL)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT article_id, title, description, source_id, source_tier
            FROM articles
            WHERE enriched_at IS NULL
            ORDER BY published_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        cols = [desc[0] for desc in conn.description]
        return [dict(zip(cols, row)) for row in rows]


def write_enrichments(
    enrichments: list[dict],
    db_path: Path | None = None,
) -> int:
    """Bulk-write enrichment fields onto existing articles.

    Each dict in ``enrichments`` should look like:
        {
            "article_id": str,
            "sentiment_score": float | None,
            "sentiment_label": str | None,
            "event_type": str | None,
            "entities": list[dict] | None,
        }

    Routings live in their own table — write them via ``write_routings``.
    """
    import json
    from datetime import datetime, timezone

    if not enrichments:
        return 0

    now = datetime.now(timezone.utc)

    with connect(db_path) as conn:
        for e in enrichments:
            conn.execute(
                """
                UPDATE articles
                SET sentiment_score = ?,
                    sentiment_label = ?,
                    event_type      = ?,
                    entities        = ?,
                    enriched_at     = ?
                WHERE article_id = ?
                """,
                [
                    e.get("sentiment_score"),
                    e.get("sentiment_label"),
                    e.get("event_type"),
                    json.dumps(e.get("entities") or []),
                    now,
                    e["article_id"],
                ],
            )
    return len(enrichments)


def write_routings(
    routings_by_article: dict[str, list[dict]],
    db_path: Path | None = None,
) -> int:
    """Write routings to the ``routings`` table.

    ``routings_by_article`` maps article_id -> list of {asset_class_id, score}.
    Existing routings for the article are deleted first (idempotent re-runs).
    """
    if not routings_by_article:
        return 0

    total_inserted = 0
    with connect(db_path) as conn:
        for article_id, routings in routings_by_article.items():
            conn.execute(
                "DELETE FROM routings WHERE article_id = ?",
                [article_id],
            )
            for r in routings:
                conn.execute(
                    """
                    INSERT INTO routings (article_id, asset_class, score)
                    VALUES (?, ?, ?)
                    """,
                    [article_id, r["asset_class_id"], r["score"]],
                )
                total_inserted += 1
    return total_inserted


def count_enriched(db_path: Path | None = None) -> tuple[int, int]:
    """Return (enriched_count, total_count) for monitoring progress."""
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE enriched_at IS NOT NULL) AS enriched,
                COUNT(*) AS total
            FROM articles
            """
        ).fetchone()
        return (row[0], row[1])