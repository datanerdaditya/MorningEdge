"""End-to-end MorningEdge pipeline runner.

This is the production entry point — invoked daily by GitHub Actions.

Steps:
    1. Init DB schema (idempotent).
    2. Fetch articles from all RSS sources (parallel, fault-tolerant).
    3. Exact dedup against existing DB (cheap, via canonical_url -> id).
    4. Fuzzy/semantic dedup against last 48h.
    5. Persist new articles.
    6. (Week 2+) Enrich, score, cluster, summarise.
    7. (Week 3+) Generate daily brief.

Until Week 2 lands, this script runs only steps 1-5.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/run_pipeline.py` from the project root,
# regardless of editable-install state.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import datetime, timezone

from loguru import logger

from morningedge.ingestion.dedup import fuzzy_dedupe
from morningedge.ingestion.rss import fetch_all
from morningedge.storage.db import (
    count_articles,
    init_schema,
    insert_articles,
)


def main() -> int:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}",
    )

    started = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {started.isoformat()}")

    # --- Step 1: schema ---
    init_schema()

    # --- Step 2: fetch ---
    before = count_articles()
    articles = fetch_all()
    if not articles:
        logger.warning("No articles fetched. Exiting.")
        return 1

    # --- Step 3 + 4: dedup ---
    # Exact dedup happens at insert time. Fuzzy dedup runs first so we
    # don't waste an insert + UPDATE cycle on near-duplicates.
    deduped = fuzzy_dedupe(articles)

    # --- Step 5: persist ---
    inserted, skipped = insert_articles(deduped)

    after = count_articles()
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    print()
    print("=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Fetched         {len(articles):>5}")
    print(f"  After fuzzy     {len(deduped):>5}  (-{len(articles) - len(deduped)} semantic dupes)")
    print(f"  Inserted        {inserted:>5}  (new)")
    print(f"  Skipped (exact) {skipped:>5}  (already in DB by URL)")
    print(f"  DB total        {after:>5}  (was {before})")
    print(f"  Duration        {duration:>5.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())