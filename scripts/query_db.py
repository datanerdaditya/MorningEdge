"""Inspect MorningEdge's enriched DB.

Usage:
    python scripts/query_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morningedge.storage.db import connect


def main() -> None:
    with connect() as conn:
        # Top-level stats
        total, enriched = conn.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE enriched_at IS NOT NULL) FROM articles"
        ).fetchone()
        print(f"\nTotal articles: {total}  |  Enriched: {enriched}")

        # Sentiment breakdown
        print("\n=== Sentiment distribution ===")
        rows = conn.execute(
            """
            SELECT sentiment_label, COUNT(*), ROUND(AVG(sentiment_score), 3)
            FROM articles
            WHERE sentiment_label IS NOT NULL
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()
        for label, count, avg in rows:
            print(f"  {label:<10} {count:>4}  (avg score {avg:+.3f})")

        # Event-type breakdown
        print("\n=== Event types ===")
        rows = conn.execute(
            """
            SELECT event_type, COUNT(*)
            FROM articles
            WHERE event_type IS NOT NULL
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()
        for event, count in rows:
            print(f"  {event:<22} {count:>4}")

        # Asset class coverage
        print("\n=== Asset class coverage ===")
        rows = conn.execute(
            """
            SELECT asset_class, COUNT(*) AS n_articles, ROUND(AVG(score), 3) AS avg_conf
            FROM routings
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()
        for cls, n, conf in rows:
            print(f"  {cls:<22} {n:>4}  (avg confidence {conf:.3f})")

        # Top sentiment-positive and negative
        print("\n=== Top 5 most BULLISH articles ===")
        rows = conn.execute(
            """
            SELECT sentiment_score, source_id, title
            FROM articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY sentiment_score DESC
            LIMIT 5
            """
        ).fetchall()
        for score, source, title in rows:
            print(f"  {score:+.2f}  [{source[:18]:<18}]  {title[:80]}")

        print("\n=== Top 5 most BEARISH articles ===")
        rows = conn.execute(
            """
            SELECT sentiment_score, source_id, title
            FROM articles
            WHERE sentiment_score IS NOT NULL
            ORDER BY sentiment_score ASC
            LIMIT 5
            """
        ).fetchall()
        for score, source, title in rows:
            print(f"  {score:+.2f}  [{source[:18]:<18}]  {title[:80]}")


if __name__ == "__main__":
    main()