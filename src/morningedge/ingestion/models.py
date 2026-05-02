"""Pydantic models shared across the ingestion layer.

This module defines the canonical shape of an "article" as it flows
through MorningEdge. Every downstream module (dedup, enrichment,
storage) consumes these types — so changing them is a contract change.

Design notes
------------
- We split RawArticle (what comes off the wire, possibly messy) from
  Article (validated, normalised, ready for the rest of the pipeline).
  This keeps the parsing logic isolated and makes failures explicit.
- We carry a stable ``article_id`` derived deterministically from the
  URL. Same URL → same id, regardless of when we ingested it. This is
  what dedup later relies on.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceTier(StrEnum):
    """Quality tier of a news source.

    Used downstream to weight aggregate sentiment scores — a Reuters
    headline shouldn't count the same as a SeekingAlpha blog post.
    """

    TIER_1 = "tier_1"  # Reuters, Bloomberg, FT, WSJ, central banks
    TIER_2 = "tier_2"  # Specialist credit pubs, S&P LCD, PDI, etc.
    TIER_3 = "tier_3"  # Aggregators, blogs, lower-signal feeds


class RawArticle(BaseModel):
    """An article as it comes off an RSS feed.

    Lenient on input — any string fields may contain junk that we'll
    clean before promoting to ``Article``. Validation here is mostly
    about presence, not content quality.
    """

    title: str
    url: str
    source_id: str  # matches a key in sources.py
    published_at: datetime | None = None
    description: str | None = None
    raw_payload: dict = Field(default_factory=dict)  # full feed entry, for debugging

    model_config = {"extra": "ignore"}


class Article(BaseModel):
    """A validated, normalised article ready for the rest of the pipeline."""

    article_id: str  # deterministic hash of canonical_url
    title: str = Field(min_length=3, max_length=500)
    url: HttpUrl
    canonical_url: str  # url with tracking params stripped
    source_id: str
    source_tier: SourceTier
    published_at: datetime
    description: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        """RSS titles often arrive with HTML entities and whitespace cruft."""
        if not isinstance(v, str):
            raise ValueError("title must be a string")
        return v.strip()

    @field_validator("published_at", "fetched_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """All timestamps in MorningEdge are stored as UTC. No exceptions."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


def make_article_id(canonical_url: str) -> str:
    """Generate a stable 16-char article id from a canonical URL.

    Deterministic: same URL always produces the same id, across runs and
    machines. This is the primary key the dedup layer keys off.
    """
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]