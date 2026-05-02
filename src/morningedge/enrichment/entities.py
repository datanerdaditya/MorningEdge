"""Entity extraction with GLiNER.

GLiNER (Generalist and Lightweight Named Entity Recognition) does
zero-shot NER: you specify the entity types you want, and it finds
them. No fine-tuning required.

For MorningEdge we extract:
    - company       : "JPMorgan Chase", "Apollo Global Management"
    - ticker        : "JPM", "AAPL", "TSLA"
    - person        : "Jamie Dimon", "Jerome Powell"
    - country       : "United States", "China"
    - money_amount  : "$5 billion", "25 basis points"
    - product       : "iPhone", "ChatGPT"

These get stored as JSON on the article row and become the foundation
for entity-level signal aggregation later (e.g. "JPM sentiment vs GS
sentiment over the last 7 days").
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

from gliner import GLiNER
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# urchade/gliner_small-v2.1 is ~150MB, fast on CPU, strong for English
# news. The base/large variants gain a few accuracy points at 2-4x cost.
MODEL_NAME = "urchade/gliner_small-v2.1"

# Entity types we care about. Order matters slightly — GLiNER scores
# each label independently but more specific labels first tends to help.
ENTITY_LABELS = [
    "company",
    "ticker",
    "person",
    "country",
    "money_amount",
    "product",
]

# Below this confidence we drop the entity (cuts noise from edge cases).
MIN_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """One extracted entity from an article."""

    text: str       # the surface form, e.g. "JPMorgan Chase"
    label: str      # one of ENTITY_LABELS
    score: float    # GLiNER's confidence, 0..1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model() -> GLiNER:
    """Load GLiNER once per process. ~150MB download on first call."""
    logger.info(f"Loading entity extraction model: {MODEL_NAME}")
    return GLiNER.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[Entity]:
    """Extract entities from a single piece of text.

    Returns a list, possibly empty. Entities below MIN_CONFIDENCE are dropped.
    """
    if not text or not text.strip():
        return []

    model = _load_model()
    raw = model.predict_entities(text, ENTITY_LABELS, threshold=MIN_CONFIDENCE)

    return [
        Entity(text=e["text"], label=e["label"], score=float(e["score"]))
        for e in raw
    ]


def extract_entities_batch(texts: list[str]) -> list[list[Entity]]:
    """Extract entities from many texts. Currently sequential; GLiNER's
    batch API is finicky and the per-call cost is already small.
    """
    return [extract_entities(t) for t in texts]