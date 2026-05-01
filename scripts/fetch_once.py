"""Run the RSS fetcher once and print a summary.

This is for manual testing — the production pipeline uses
``scripts/run_pipeline.py`` instead.

Usage:
    python scripts/fetch_once.py
"""

from __future__ import annotations

import sys
from collections import Counter

from loguru import logger

from morningedge.ingestion.rss import fetch_all


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}")

    articles = fetch_all()

    print()
    print("=" * 70)
    print(f"  Fetched {len(articles)} articles")
    print("=" * 70)

    by_source = Counter(a.source_id for a in articles)
    for source_id, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source_id:<30} {count:>4}")

    print()
    print("Sample headlines:")
    print("-" * 70)
    for a in articles[:8]:
        ts = a.published_at.strftime("%Y-%m-%d %H:%M")
        print(f"  [{ts}] [{a.source_id}]")
        print(f"    {a.title[:100]}")
        print(f"    {a.canonical_url}")
        print()


if __name__ == "__main__":
    main()