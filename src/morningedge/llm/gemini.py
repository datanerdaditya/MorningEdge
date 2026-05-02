"""Gemini client with model routing.

Three Gemini models, each with different free-tier daily quotas:
    Flash-Lite : 1,000 RPD — bulk summarisation, theme labels
    Flash      :   250 RPD — top-story sentiment re-scoring
    Pro        :   100 RPD — daily brief, complex chat queries

This module is the single chokepoint for all Gemini calls. It tracks
in-process quota usage and applies the right model per task type.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from threading import Lock

from google import genai
from loguru import logger

from morningedge.config import settings

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro"

# Free-tier per-minute limits, with a small safety margin.
# Updated to actual values surfaced by Gemini's 429 responses; if a
# limit changes, lower these by 1 to be safe.
RPM_LIMITS: dict[str, int] = {
    MODEL_FLASH_LITE: 14,  # actual cap 15, leave headroom
    MODEL_FLASH: 4,        # actual cap 5, leave headroom
    MODEL_PRO: 4,          # actual cap 5, leave headroom
}

# Tracks call timestamps per model for rate limiting.
_call_history: dict[str, deque] = {m: deque() for m in RPM_LIMITS}
_rate_lock = Lock()


def _wait_for_quota(model: str) -> None:
    """Block until we can make another call to this model.

    Sliding-window rate limiter: keeps timestamps of the last N calls
    where N is the RPM limit, and waits if the oldest is less than
    60 seconds ago.
    """
    if model not in RPM_LIMITS:
        return  # not tracked

    limit = RPM_LIMITS[model]
    history = _call_history[model]

    with _rate_lock:
        now = time.monotonic()
        # Drop timestamps older than 60s
        while history and now - history[0] > 60:
            history.popleft()

        if len(history) >= limit:
            sleep_for = 60 - (now - history[0]) + 0.5  # small buffer
            logger.info(f"Rate limit on {model}: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            # After sleeping, drop expired again
            now = time.monotonic()
            while history and now - history[0] > 60:
                history.popleft()

        history.append(time.monotonic())


# ---------------------------------------------------------------------------
# In-process quota tracker
# ---------------------------------------------------------------------------


@dataclass
class _QuotaState:
    today: date
    by_model: dict[str, int]


_quota = _QuotaState(today=date.today(), by_model={})


def _bump_quota(model: str) -> int:
    """Increment today's call count for a model. Returns new count."""
    today = date.today()
    if today != _quota.today:
        _quota.today = today
        _quota.by_model = {}
    _quota.by_model[model] = _quota.by_model.get(model, 0) + 1
    return _quota.by_model[model]


def quota_summary() -> dict[str, int]:
    """Return today's calls per model."""
    return dict(_quota.by_model)


# ---------------------------------------------------------------------------
# Client (lazy)
# ---------------------------------------------------------------------------


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_gemini(
    prompt: str,
    model: str = MODEL_FLASH_LITE,
    response_json: bool = False,
) -> str:
    """Call Gemini and return the text response.

    Set ``response_json=True`` to force JSON-mode output (lower
    hallucination, easier downstream parsing).
    """
    client = _get_client()

    config = None
    if response_json:
        config = {"response_mime_type": "application/json"}

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        logger.warning(f"Gemini call failed ({model}): {e}")
        raise

    count = _bump_quota(model)
    logger.debug(f"Gemini {model} call #{count} today")

    return response.text or ""


# ---------------------------------------------------------------------------
# Top-story sentiment rescore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeminiSentiment:
    score: float           # [-1, +1]
    label: str             # 'positive' | 'neutral' | 'negative'
    reasoning: str


def rescore_sentiment(title: str, description: str | None) -> GeminiSentiment | None:
    """Use Gemini Flash to rescore one article's sentiment.

    Returns None if the model produced unparseable output.
    """
    prompt = _RESCORE_PROMPT.format(
        title=title,
        description=description or "(no description provided)",
    )

    raw = call_gemini(prompt, model=MODEL_FLASH, response_json=True)

    try:
        data = json.loads(raw)
        score = float(data["score"])
        label = data["label"].lower()
        reasoning = data.get("reasoning", "")
        if label not in {"positive", "neutral", "negative"}:
            label = "neutral"
        return GeminiSentiment(
            score=max(-1.0, min(1.0, score)),
            label=label,
            reasoning=reasoning,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Gemini rescore parse failed: {e}; raw={raw[:200]}")
        return None


_RESCORE_PROMPT = """\
You are a financial sentiment classifier specialising in credit and macro markets.
Read this news headline and return a JSON object with three fields:
- "score": a float from -1.0 (very bearish for the issuer/asset) to +1.0 (very bullish)
- "label": one of "positive", "neutral", "negative"
- "reasoning": ONE short sentence explaining your call

Important context for scoring:
- "Beats earnings", "raises guidance", "tightening spreads" -> positive
- "Default", "downgrade", "widening spreads", "Chapter 11" -> very negative
- "Hawkish Fed", "rate hike" -> negative for risk assets, positive for the dollar
- "Dovish Fed", "rate cut" -> positive for risk assets, negative for the dollar
- Take "blowout earnings" as POSITIVE (finance idiom: very strong beat).
- Be skeptical of routine procedural news (Form 144 filings, scheduled meetings) -> typically neutral.

Title: {title}
Description: {description}

Return ONLY the JSON object, no other text.
"""