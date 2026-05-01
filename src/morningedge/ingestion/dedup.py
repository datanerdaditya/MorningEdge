"""Article deduplication — exact and fuzzy.

The pipeline runs two passes of dedup:

1. **Exact** dedup happens at insert time in ``storage.db.insert_articles``
   — same canonical URL → same article_id → silently skipped on insert.

2. **Fuzzy / semantic** dedup happens here, before insert. Identifies
   syndicated stories where multiple outlets ran the same wire copy
   with slightly different headlines and different URLs. Uses
   sentence-transformer embeddings + cosine similarity.

Why split the two passes? Exact dedup is essentially free. Fuzzy dedup
costs ~50ms per batch (loading the model) plus ~5ms per article.
Doing them in the right order (exact first via DB constraints, fuzzy
second on the remaining new articles) keeps the pipeline fast.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from morningedge.ingestion.models import Article
from morningedge.storage.db import connect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2: 384 dim, 80MB, fast, good baseline for news headlines.
# Stronger options (mpnet-base, ~420MB) gain ~1-2% accuracy at 4x cost.
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Cosine similarity above this means "same story".
# Calibrated empirically on financial news headlines:
#   0.95+ : essentially verbatim, rare
#   0.92  : same wire, edited headlines  ← chosen threshold
#   0.85  : same topic, related angle (don't dedup)
SIMILARITY_THRESHOLD = 0.85

# How far back to look for potential duplicates. News cycles are short.
LOOKBACK_HOURS = 48


# ---------------------------------------------------------------------------
# Model loading (lazy + cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once per process."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


EMBEDDING_DIM = 768  # for all-mpnet-base-v2; was 384 for MiniLM


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts as L2-normalised embedding vectors.

    Returns a (len(texts), EMBEDDING_DIM) numpy array. Normalised so cosine
    similarity becomes a dot product.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# Recent-articles lookup
# ---------------------------------------------------------------------------


def _recent_titles_with_embeddings(
    hours: int = LOOKBACK_HOURS,
) -> tuple[list[str], np.ndarray]:
    """Pull the last N hours of titles and their cached embeddings.

    For articles without a cached embedding (fresh DB), we compute on
    the fly. Future runs will hit the cache.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT article_id, title, embedding
            FROM articles
            WHERE fetched_at >= ?
            """,
            [cutoff],
        ).fetchall()

    if not rows:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    
    ids, titles, raw_embeddings = zip(*rows)

    # Some rows may have NULL embeddings (older articles, pre-Day 4 data).
    # We embed those on demand and persist for next time.
    needs_embed = [(i, t) for i, (rid, t, e) in enumerate(zip(ids, titles, raw_embeddings)) if e is None]
    cached = [_decode_embedding(e) if e else None for e in raw_embeddings]

    if needs_embed:
        new_vecs = embed_texts([t for _, t in needs_embed])
        for (i, _), vec in zip(needs_embed, new_vecs):
            cached[i] = vec
        # Persist them so we don't re-embed next run
        _save_embeddings(
            [ids[i] for i, _ in needs_embed],
            [cached[i] for i, _ in needs_embed],
        )

    matrix = np.vstack(cached) if cached else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return list(titles), matrix


def _decode_embedding(json_value: list) -> np.ndarray:
    """DuckDB returns JSON as a Python list — convert to a numpy vector."""
    return np.asarray(json_value, dtype=np.float32)


def _save_embeddings(article_ids: list[str], vectors: list[np.ndarray]) -> None:
    """Persist computed embeddings back to the DB (as JSON arrays)."""
    if not article_ids:
        return
    import json
    with connect() as conn:
        for aid, vec in zip(article_ids, vectors):
            conn.execute(
                "UPDATE articles SET embedding = ? WHERE article_id = ?",
                [json.dumps(vec.tolist()), aid],
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fuzzy_dedupe(articles: list[Article]) -> list[Article]:
    """Filter out articles that are semantic duplicates of recent ones.

    Strategy:
        1. Embed all incoming candidate titles in one batch (fast).
        2. Pull recent existing titles + their embeddings.
        3. Compute cosine similarities (matrix mul).
        4. Drop any candidate with max similarity >= threshold.
        5. Within the kept candidates, also dedupe against each other.
    """
    if not articles:
        return []

    candidate_titles = [a.title for a in articles]
    candidate_vecs = embed_texts(candidate_titles)

    existing_titles, existing_vecs = _recent_titles_with_embeddings()

    keep: list[Article] = []
    keep_vecs: list[np.ndarray] = []
    dropped = 0

    for article, vec in zip(articles, candidate_vecs):
        # Compare to existing DB articles
        if existing_vecs.shape[0] > 0:
            sims_existing = existing_vecs @ vec
            if float(sims_existing.max()) >= SIMILARITY_THRESHOLD:
                dropped += 1
                continue

        # Compare to other candidates we've already chosen to keep
        if keep_vecs:
            sims_keep = np.array([float(v @ vec) for v in keep_vecs])
            if sims_keep.max() >= SIMILARITY_THRESHOLD:
                dropped += 1
                continue

        keep.append(article)
        keep_vecs.append(vec)

    if dropped:
        logger.info(f"Fuzzy dedup: dropped {dropped} of {len(articles)} as semantic duplicates")
    return keep