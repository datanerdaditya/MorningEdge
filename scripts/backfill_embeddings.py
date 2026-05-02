"""One-shot script to compute and persist embeddings for all articles
that don't have them yet. Required so the RAG chat can retrieve them.

Usage:
    python scripts/backfill_embeddings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.ingestion.dedup import embed_texts
from morningedge.storage.db import connect


BATCH_SIZE = 64


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT article_id, title, description
            FROM articles
            WHERE embedding IS NULL
            ORDER BY published_at DESC
            """
        ).fetchall()

    if not rows:
        logger.info("All articles already have embeddings.")
        return

    logger.info(f"Backfilling embeddings for {len(rows)} articles")

    # Embed in batches
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        texts = [
            f"{title}. {desc}" if desc else title
            for _, title, desc in batch
        ]
        vectors = embed_texts(texts)

        with connect() as conn:
            for (article_id, _, _), vec in zip(batch, vectors):
                conn.execute(
                    "UPDATE articles SET embedding = ? WHERE article_id = ?",
                    [json.dumps(vec.tolist()), article_id],
                )

        logger.info(f"  Persisted {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)}")

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()