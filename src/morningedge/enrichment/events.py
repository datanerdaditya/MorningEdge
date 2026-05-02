"""Event classification via zero-shot NLI.

We use a BART-MNLI model to classify each article into one of our
finance-specific event types. Zero-shot: no fine-tuning needed.

The event taxonomy is opinionated. Each label is phrased as a short
description that BART's NLI head can score against the article text:
    "This article is about an earnings announcement"
    "This article is about a corporate default"

We pick the highest-scoring label, but only commit to it if its
confidence clears MIN_CONFIDENCE. Otherwise we tag as 'other'.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DeBERTa-v3-base trained on MNLI/FEVER/ANLI. ~350MB, fast, stable on Apple
# Silicon. Originally tried BART-large-MNLI (1.6GB) but it caused bus errors
# on M-series Macs due to memory pressure during attention computation.
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Event taxonomy. The strings on the right are the actual hypotheses
# fed to the NLI model — natural-language sentences work much better
# than bare labels.
EVENT_TYPES: dict[str, str] = {
    "earnings":           "This article is about a company's earnings announcement or financial results",
    "m_and_a":            "This article is about a merger, acquisition, or buyout",
    "central_bank":       "This article is about a central bank decision, monetary policy, or interest rates",
    "default_distress":   "This article is about a corporate default, bankruptcy, or distressed debt",
    "ratings_change":     "This article is about a credit rating upgrade or downgrade",
    "regulatory":         "This article is about regulation, government action, or legal proceedings",
    "fundraising":        "This article is about a company raising capital, issuing bonds, or fund launches",
    "executive_change":   "This article is about an executive appointment, resignation, or departure",
    "macro_data":         "This article is about economic data such as inflation, GDP, jobs, or PMI",
    "product_launch":     "This article is about a new product or technology launch",
    "geopolitical":       "This article is about geopolitics, trade, or international conflict",
    "other":              "This article is general financial news",
}

MIN_CONFIDENCE = 0.25


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventResult:
    """The most likely event type for an article."""

    event_type: str   # key from EVENT_TYPES
    score: float      # confidence, 0..1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_classifier():
    """Load the zero-shot pipeline once per process.

    We pin to CPU explicitly. On Apple Silicon, transformers can auto-select
    the MPS (Metal) backend, which is unstable for large attention models
    and has caused bus errors in the past. CPU is slower but reliable.
    """
    logger.info(f"Loading event classifier: {MODEL_NAME}")
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,  # -1 = CPU; do NOT auto-select MPS on Apple Silicon
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Minimum gap between top-1 and top-2 label scores. A "confident" pick
# is one where the model clearly prefers one label over the runner-up.
MIN_MARGIN = 0.05


def classify_event(text: str) -> EventResult:
    """Classify a single article's event type.

    A classification is accepted if BOTH:
      - top score >= MIN_CONFIDENCE (model is confident in absolute terms)
      - margin (top - second) >= MIN_MARGIN (model is confident relatively)

    Returns ``EventResult(event_type='other', score=...)`` otherwise.
    """
    if not text or not text.strip():
        return EventResult(event_type="other", score=0.0)

    classifier = _load_classifier()
    hypotheses = list(EVENT_TYPES.values())

    result = classifier(
        text,
        candidate_labels=hypotheses,
        multi_label=False,
    )

    scores = result["scores"]
    top_score = float(scores[0])
    margin = top_score - float(scores[1]) if len(scores) > 1 else top_score

    top_hypothesis = result["labels"][0]
    event_type = next(
        (k for k, v in EVENT_TYPES.items() if v == top_hypothesis),
        "other",
    )

    if top_score < MIN_CONFIDENCE or margin < MIN_MARGIN:
        return EventResult(event_type="other", score=top_score)

    return EventResult(event_type=event_type, score=top_score)


def classify_events_batch(texts: list[str]) -> list[EventResult]:
    """Classify many texts. Sequential — BART-MNLI is the slow link in
    the pipeline so we keep it simple; parallel inference would need
    careful batching against memory.
    """
    return [classify_event(t) for t in texts]