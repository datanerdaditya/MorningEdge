"""Score a few articles from the DB with FinBERT to sanity-check the sentiment layer.

Usage:
    python scripts/score_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.enrichment.sentiment import score_articles_batch
from morningedge.storage.db import connect


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    # Pull a sample of recent articles
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT title, description, source_id
            FROM articles
            ORDER BY published_at DESC
            LIMIT 20
            """
        ).fetchall()

    if not rows:
        print("No articles in DB. Run `python scripts/run_pipeline.py` first.")
        return

    pairs = [(r[0], r[1]) for r in rows]
    sources = [r[2] for r in rows]

    results = score_articles_batch(pairs)

    print()
    print("=" * 100)
    print(f"  {'SCORE':>6}  {'LABEL':<10} {'SOURCE':<22} TITLE")
    print("=" * 100)
    for (title, _), source, result in sorted(
        zip(pairs, sources, results),
        key=lambda x: x[2].score,
    ):
        marker = "🟢" if result.score > 0.2 else ("🔴" if result.score < -0.2 else "⚪")
        print(f"  {marker} {result.score:+.2f}  {result.label:<10} {source:<22} {title[:70]}")


if __name__ == "__main__":
    main()