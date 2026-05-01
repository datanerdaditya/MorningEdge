"""Run the RSS fetcher once and persist results to DuckDB.

Usage:
    python -m scripts.fetch_once
"""

from __future__ import annotations

import sys
from collections import Counter

from loguru import logger

from morningedge.ingestion.rss import fetch_all
from morningedge.storage.db import (
    count_articles,
    init_schema,
    insert_articles,
    recent_articles,
)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}",
    )

    init_schema()

    before = count_articles()

    articles = fetch_all()
    inserted, skipped = insert_articles(articles)

    after = count_articles()

    print()
    print("=" * 70)
    print(f"  Fetched   {len(articles):>5}")
    print(f"  Inserted  {inserted:>5}  (new)")
    print(f"  Skipped   {skipped:>5}  (already in DB)")
    print(f"  DB total  {after:>5}  (was {before} before this run)")
    print("=" * 70)

    by_source = Counter(a.source_id for a in articles)
    for source_id, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source_id:<30} {count:>4}")

    print()
    print("Most recent in DB:")
    print("-" * 70)
    for row in recent_articles(limit=5):
        ts = row["published_at"].strftime("%Y-%m-%d %H:%M")
        print(f"  [{ts}] [{row['source_id']}]")
        print(f"    {row['title'][:100]}")
        print()


if __name__ == "__main__":
    main()