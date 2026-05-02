"""Read-only queries for the dashboard.

Keeps SQL out of UI code. Every function returns either a pandas
DataFrame or a plain dict — easy to render in Streamlit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from morningedge.storage.db import connect
from morningedge.taxonomy import TAXONOMY

# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def overall_summary(days_back: int = 1) -> dict:
    """High-level numbers for the regime banner."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(sentiment_score) AS avg_sentiment,
                COUNT(*) FILTER (WHERE sentiment_label = 'positive') AS n_pos,
                COUNT(*) FILTER (WHERE sentiment_label = 'negative') AS n_neg
            FROM articles
            WHERE enriched_at IS NOT NULL
              AND published_at >= ?
            """,
            [cutoff],
        ).fetchone()

    n, avg, n_pos, n_neg = row
    return {
        "n_articles": int(n or 0),
        "avg_sentiment": float(avg or 0.0),
        "n_positive": int(n_pos or 0),
        "n_negative": int(n_neg or 0),
    }


# ---------------------------------------------------------------------------
# Per-asset-class summaries
# ---------------------------------------------------------------------------


def asset_class_summary(asset_class_id: str, days_back: int = 1) -> dict:
    """Aggregate stats for one asset class."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(a.sentiment_score) AS avg_sentiment
            FROM articles a
            JOIN routings r ON a.article_id = r.article_id
            WHERE r.asset_class = ?
              AND a.enriched_at IS NOT NULL
              AND a.published_at >= ?
            """,
            [asset_class_id, cutoff],
        ).fetchone()

    n, avg = row
    return {
        "asset_class_id": asset_class_id,
        "n_articles": int(n or 0),
        "avg_sentiment": float(avg or 0.0),
    }


def all_asset_class_summaries(days_back: int = 1) -> list[dict]:
    """Summaries for every class in the taxonomy."""
    return [
        {**asset_class_summary(ac.id, days_back), "label": ac.label, "tier": ac.tier}
        for ac in TAXONOMY
    ]


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------


def latest_narratives(asset_class_id: str | None = None, limit: int = 20) -> pd.DataFrame:
    """Most recent narratives, optionally filtered to one asset class."""
    with connect() as conn:
        if asset_class_id:
            rows = conn.execute(
                """
                SELECT narrative_date, asset_class, title, summary, article_count
                FROM narratives
                WHERE asset_class = ?
                ORDER BY narrative_date DESC, article_count DESC
                LIMIT ?
                """,
                [asset_class_id, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT narrative_date, asset_class, title, summary, article_count
                FROM narratives
                ORDER BY narrative_date DESC, article_count DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)


def top_narrative_for_class(asset_class_id: str) -> dict | None:
    """The single most-cited narrative for an asset class, or None."""
    df = latest_narratives(asset_class_id=asset_class_id, limit=1)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


def articles_for_class(asset_class_id: str, limit: int = 50, days_back: int = 2) -> pd.DataFrame:
    """Recent articles routed to one asset class, sorted by sentiment magnitude."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.title, a.source_id, a.published_at, a.canonical_url,
                   a.sentiment_score, a.sentiment_label, a.event_type,
                   r.score AS routing_score
            FROM articles a
            JOIN routings r ON a.article_id = r.article_id
            WHERE r.asset_class = ?
              AND a.published_at >= ?
            ORDER BY ABS(a.sentiment_score) DESC NULLS LAST
            LIMIT ?
            """,
            [asset_class_id, cutoff, limit],
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def regime_label(avg_sentiment: float) -> str:
    """Convert a continuous score to a human-readable regime label."""
    if avg_sentiment > 0.15:
        return "Risk-On"
    if avg_sentiment < -0.15:
        return "Risk-Off"
    return "Mixed"

# ---------------------------------------------------------------------------
# Drill-down queries (Day 11)
# ---------------------------------------------------------------------------


def sentiment_timeline_for_class(asset_class_id: str, days_back: int = 7) -> pd.DataFrame:
    """Daily average sentiment for one asset class. Used for the line chart."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DATE(a.published_at) AS day,
                   AVG(a.sentiment_score) AS avg_sentiment,
                   COUNT(*) AS n_articles
            FROM articles a
            JOIN routings r ON a.article_id = r.article_id
            WHERE r.asset_class = ?
              AND a.enriched_at IS NOT NULL
              AND a.published_at >= ?
            GROUP BY 1
            ORDER BY 1
            """,
            [asset_class_id, cutoff],
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)


def top_entities_for_class(
    asset_class_id: str,
    days_back: int = 2,
    label_filter: list[str] | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Most-mentioned entities in articles routed to one asset class.

    DuckDB lets us crack the JSON 'entities' column inline. We expand
    each entity to its own row, then group + count.
    """
    cutoff = datetime.now(UTC) - timedelta(days=days_back)
    label_clause = ""
    params = [asset_class_id, cutoff]
    if label_filter:
        placeholders = ",".join("?" * len(label_filter))
        label_clause = f"AND label IN ({placeholders})"
        params.extend(label_filter)
    params.append(limit)

    with connect() as conn:
        rows = conn.execute(
            f"""
            WITH expanded AS (
                SELECT a.article_id,
                       a.sentiment_score,
                       UNNEST(json_extract(a.entities, '$[*]')) AS ent_json
                FROM articles a
                JOIN routings r ON a.article_id = r.article_id
                WHERE r.asset_class = ?
                  AND a.entities IS NOT NULL
                  AND a.published_at >= ?
            ),
            parsed AS (
                SELECT article_id,
                       sentiment_score,
                       json_extract_string(ent_json, '$.text')  AS entity_text,
                       json_extract_string(ent_json, '$.label') AS label
                FROM expanded
            )
            SELECT entity_text AS entity,
                   label,
                   COUNT(*) AS mentions,
                   ROUND(AVG(sentiment_score), 3) AS avg_sentiment
            FROM parsed
            WHERE entity_text IS NOT NULL
              {label_clause}
            GROUP BY 1, 2
            ORDER BY mentions DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)


def event_breakdown_for_class(asset_class_id: str, days_back: int = 2) -> pd.DataFrame:
    """Distribution of event types within one asset class."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT a.event_type,
                   COUNT(*) AS n,
                   ROUND(AVG(a.sentiment_score), 3) AS avg_sentiment
            FROM articles a
            JOIN routings r ON a.article_id = r.article_id
            WHERE r.asset_class = ?
              AND a.event_type IS NOT NULL
              AND a.published_at >= ?
            GROUP BY 1
            ORDER BY n DESC
            """,
            [asset_class_id, cutoff],
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)


def global_top_entities(
    days_back: int = 2,
    label_filter: list[str] | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    """Top entities across all asset classes."""
    cutoff = datetime.now(UTC) - timedelta(days=days_back)
    label_clause = ""
    params: list = [cutoff]
    if label_filter:
        placeholders = ",".join("?" * len(label_filter))
        label_clause = f"AND label IN ({placeholders})"
        params.extend(label_filter)
    params.append(limit)

    with connect() as conn:
        rows = conn.execute(
            f"""
            WITH expanded AS (
                SELECT article_id,
                       sentiment_score,
                       UNNEST(json_extract(entities, '$[*]')) AS ent_json
                FROM articles
                WHERE entities IS NOT NULL
                  AND published_at >= ?
            ),
            parsed AS (
                SELECT article_id,
                       sentiment_score,
                       json_extract_string(ent_json, '$.text')  AS entity_text,
                       json_extract_string(ent_json, '$.label') AS label
                FROM expanded
            )
            SELECT entity_text AS entity,
                   label,
                   COUNT(*) AS mentions,
                   ROUND(AVG(sentiment_score), 3) AS avg_sentiment
            FROM parsed
            WHERE entity_text IS NOT NULL
              {label_clause}
            GROUP BY 1, 2
            ORDER BY mentions DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        cols = [d[0] for d in conn.description]
        return pd.DataFrame(rows, columns=cols)