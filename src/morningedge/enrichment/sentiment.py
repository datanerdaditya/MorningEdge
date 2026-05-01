"""Sentiment scoring with FinBERT.

The bulk sentiment layer: every article gets a fast, local sentiment
score from FinBERT (ProsusAI/finbert). The Gemini re-scoring pass for
top stories happens separately in ``llm/`` and runs on far fewer rows.

Design notes
------------
- Score format: continuous [-1.0, +1.0] where +1 is strongly bullish
  and -1 is strongly bearish. Computed as P(positive) - P(negative).
- Title + description both feed the model, weighted 70/30. Description
  text often disambiguates tone.
- Batched inference: 32 articles per forward pass, ~10x faster than
  sequential.
- Fail-soft: bad inputs return NULL sentiment rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_TOKENS = 256  # FinBERT max is 512; 256 is plenty for title+desc
TITLE_WEIGHT = 0.7
DESC_WEIGHT = 0.3

# FinBERT label order in its config: [positive, negative, neutral]
# Verified at https://huggingface.co/ProsusAI/finbert/blob/main/config.json
LABEL_ORDER = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentResult:
    """Output of FinBERT sentiment scoring for one article."""

    score: float            # [-1, +1], = P(pos) - P(neg)
    label: str              # 'positive' | 'neutral' | 'negative'
    p_positive: float
    p_neutral: float
    p_negative: float

    @classmethod
    def null(cls) -> "SentimentResult":
        """Sentinel returned when scoring fails for an article."""
        return cls(0.0, "neutral", 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Model loading (lazy + cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load FinBERT and tokenizer once per process. Idempotent."""
    logger.info(f"Loading sentiment model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # inference mode — disables dropout, etc.
    return model, tokenizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_texts(texts: list[str]) -> list[SentimentResult]:
    """Score a list of strings with FinBERT.

    Returns one SentimentResult per input, in order. Empty/None inputs
    get a NULL result (neutral, score=0).
    """
    if not texts:
        return []

    model, tokenizer = _load_model()
    results: list[SentimentResult] = [SentimentResult.null()] * len(texts)

    # Find non-empty indices to actually score
    valid_pairs = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not valid_pairs:
        return results

    indices, valid_texts = zip(*valid_pairs)

    # Batch through the model
    with torch.no_grad():
        for batch_start in range(0, len(valid_texts), BATCH_SIZE):
            batch = list(valid_texts[batch_start : batch_start + BATCH_SIZE])
            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_TOKENS,
                )
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).numpy()
            except Exception as e:
                logger.warning(f"FinBERT batch failed: {e}")
                continue

            for offset, prob_row in enumerate(probs):
                idx = indices[batch_start + offset]
                results[idx] = _build_result(prob_row)

    return results


def score_article(title: str, description: str | None) -> SentimentResult:
    """Score a single article using title + description (weighted)."""
    title_res = score_texts([title])[0] if title else SentimentResult.null()

    if not description or not description.strip():
        return title_res

    desc_res = score_texts([description])[0]

    # Weighted blend of probabilities
    p_pos = TITLE_WEIGHT * title_res.p_positive + DESC_WEIGHT * desc_res.p_positive
    p_neg = TITLE_WEIGHT * title_res.p_negative + DESC_WEIGHT * desc_res.p_negative
    p_neu = TITLE_WEIGHT * title_res.p_neutral + DESC_WEIGHT * desc_res.p_neutral

    return _build_result(np.array([p_pos, p_neg, p_neu]))


def score_articles_batch(
    articles: list[tuple[str, str | None]],
) -> list[SentimentResult]:
    """Score many (title, description) pairs efficiently.

    Optimised version of calling ``score_article`` in a loop: we collect
    all titles and all descriptions, batch-score each, then blend.
    """
    if not articles:
        return []

    titles = [a[0] for a in articles]
    descriptions = [a[1] or "" for a in articles]

    title_results = score_texts(titles)
    # Only score non-empty descriptions to save compute
    desc_indices = [i for i, d in enumerate(descriptions) if d.strip()]
    desc_results_partial = score_texts([descriptions[i] for i in desc_indices])
    desc_results: list[SentimentResult | None] = [None] * len(articles)
    for i, idx in enumerate(desc_indices):
        desc_results[idx] = desc_results_partial[i]

    blended: list[SentimentResult] = []
    for tr, dr in zip(title_results, desc_results):
        if dr is None:
            blended.append(tr)
        else:
            p_pos = TITLE_WEIGHT * tr.p_positive + DESC_WEIGHT * dr.p_positive
            p_neg = TITLE_WEIGHT * tr.p_negative + DESC_WEIGHT * dr.p_negative
            p_neu = TITLE_WEIGHT * tr.p_neutral + DESC_WEIGHT * dr.p_neutral
            blended.append(_build_result(np.array([p_pos, p_neg, p_neu])))

    return blended


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_result(probs: np.ndarray) -> SentimentResult:
    """Convert a [P(pos), P(neg), P(neu)] vector into a SentimentResult."""
    p_pos, p_neg, p_neu = float(probs[0]), float(probs[1]), float(probs[2])
    score = p_pos - p_neg
    label = LABEL_ORDER[int(np.argmax(probs))]
    return SentimentResult(
        score=score,
        label=label,
        p_positive=p_pos,
        p_neutral=p_neu,
        p_negative=p_neg,
    )