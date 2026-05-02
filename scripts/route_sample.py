"""Run the asset-class router against recent DB articles to sanity-check.

Usage:
    python scripts/route_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.enrichment.router import route_text
from morningedge.storage.db import connect


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT title, description, source_id
            FROM articles
            ORDER BY published_at DESC
            LIMIT 30
            """
        ).fetchall()

    if not rows:
        print("No articles in DB. Run the pipeline first.")
        return

    print()
    print("=" * 110)
    print(f"  {'SOURCE':<22} TITLE")
    print(f"  {'':<22} ROUTED TO")
    print("=" * 110)

    for title, description, source in rows:
        text = title + ". " + (description or "")
        routings = route_text(text)

        print(f"  {source:<22} {title[:80]}")
        if not routings:
            print(f"  {'':<22} → (no confident routing)")
        else:
            tags = "  ".join(f"{r.asset_class_id} ({r.score:.2f})" for r in routings)
            print(f"  {'':<22} → {tags}")
        print()


if __name__ == "__main__":
    main()