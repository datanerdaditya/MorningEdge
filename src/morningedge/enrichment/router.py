"""Asset-class routing via semantic similarity.

Replaces brittle keyword-query routing with mpnet embedding similarity:
each article is matched against the asset-class descriptions in
``taxonomy.py``. Multi-label: an article can belong to several classes
(e.g. a Fed decision is both 'rates' and 'us_macro').

Why this is better than keywords
--------------------------------
- "Apple raises $5bn in bond market" → keyword router sends this to
  'tech'. Embedding router sends it to 'high_yield' + 'tech'. Correct.
- "Apollo launches new direct lending fund" → keyword router has no
  idea. Embedding router routes to 'private_credit'. Correct.
- "Fed signals more cuts ahead" → keyword router sends to 'rates'.
  Embedding router routes to 'rates' + 'us_macro' + 'banks'. Correct.

Reused embedding model
----------------------
We use the same sentence-transformer model (all-mpnet-base-v2) loaded
by ``ingestion.dedup``. The pipeline computes each article's embedding
exactly once and reuses it across dedup, routing, and clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from loguru import logger

from morningedge.ingestion.dedup import embed_texts
from morningedge.taxonomy import TAXONOMY, AssetClass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Articles must clear this similarity to be assigned to an asset class.
# Calibrated on the test cases at the bottom of this module — articles
# that clearly belong to a class score 0.40+, ambiguous ones 0.30-0.40,
# unrelated ones below 0.30.
ROUTING_THRESHOLD = 0.40

# Cap on how many asset classes one article can route to. Prevents
# noise from over-classification.
MAX_ASSIGNMENTS = 3


# ---------------------------------------------------------------------------
# Asset class embeddings (computed once, cached forever)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _asset_class_embeddings() -> tuple[list[AssetClass], np.ndarray]:
    """Embed all asset class descriptions. Cached for the process lifetime."""
    logger.info(f"Computing embeddings for {len(TAXONOMY)} asset classes")
    descriptions = [ac.description for ac in TAXONOMY]
    matrix = embed_texts(descriptions)
    return list(TAXONOMY), matrix


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Routing:
    """One article-to-asset-class assignment with a confidence score."""

    asset_class_id: str
    score: float  # cosine similarity, 0..1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_text(text: str) -> list[Routing]:
    """Route a single piece of text to its top asset classes.

    Returns a list (possibly empty) of Routings, sorted by score
    descending. Capped at MAX_ASSIGNMENTS, filtered by ROUTING_THRESHOLD.
    """
    if not text or not text.strip():
        return []

    classes, class_matrix = _asset_class_embeddings()
    article_vec = embed_texts([text])[0]

    sims = class_matrix @ article_vec  # cosine sim (vectors already L2-normalised)
    return _select_routings(classes, sims)


def route_texts(texts: list[str]) -> list[list[Routing]]:
    """Route many texts at once. More efficient than calling route_text in a loop."""
    if not texts:
        return []

    classes, class_matrix = _asset_class_embeddings()
    article_vecs = embed_texts(texts)

    # (n_articles, n_classes) similarity matrix
    sim_matrix = article_vecs @ class_matrix.T

    return [_select_routings(classes, sim_matrix[i]) for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_routings(
    classes: list[AssetClass],
    similarities: np.ndarray,
) -> list[Routing]:
    """Filter by threshold, sort, cap at top K."""
    candidates = [
        Routing(asset_class_id=ac.id, score=float(sim))
        for ac, sim in zip(classes, similarities, strict=False)
        if float(sim) >= ROUTING_THRESHOLD
    ]
    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:MAX_ASSIGNMENTS]