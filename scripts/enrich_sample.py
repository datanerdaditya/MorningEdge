"""Run sentiment + routing + entities + events on recent DB articles.

Shows the full enrichment pipeline (Day 5 + 6 + 7) on real headlines.

Usage:
    python scripts/enrich_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.enrichment.entities import extract_entities
from morningedge.enrichment.events import classify_event
from morningedge.enrichment.router import route_text
from morningedge.enrichment.sentiment import score_article
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
            LIMIT 10
            """
        ).fetchall()

    if not rows:
        print("No articles. Run the pipeline first.")
        return

    print()
    for title, desc, source in rows:
        text = title + ". " + (desc or "")

        sentiment = score_article(title, desc)
        routings = route_text(text)
        entities = extract_entities(text)
        event = classify_event(text)

        print("=" * 100)
        print(f"  [{source}]  {title[:90]}")
        print(f"  Sentiment   {sentiment.score:+.2f}  ({sentiment.label})")
        print(f"  Event       {event.event_type}  ({event.score:.2f})")
        if routings:
            tags = "  ".join(f"{r.asset_class_id}({r.score:.2f})" for r in routings)
            print(f"  Routing     {tags}")
        else:
            print(f"  Routing     —")
        if entities:
            ent_summary = ", ".join(f"{e.text}[{e.label}]" for e in entities[:5])
            print(f"  Entities    {ent_summary}")
        else:
            print(f"  Entities    —")
        print()


if __name__ == "__main__":
    main()