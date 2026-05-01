"""RSS feed ingestion.

Pulls from the curated source registry in ``sources.py`` and returns
validated ``Article`` objects ready for the rest of the pipeline.

Design notes
------------
- Async parallel fetch via httpx + asyncio. One slow source can't hold
  up the others.
- Per-source isolation: a feed that 500s, times out, or returns garbage
  is logged and skipped. The pipeline never crashes because Reuters
  decided to have a bad day.
- URL canonicalisation strips known tracking params so the same article
  syndicated with different ``utm_source`` values produces the same id.
- Retries via tenacity with exponential backoff (3 attempts, 1s/2s/4s).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import httpx
from dateutil import parser as dateparser
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from morningedge.ingestion.models import (
    Article,
    RawArticle,
    SourceTier,
    make_article_id,
)
from morningedge.ingestion.sources import Source, all_sources

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default UA for most sites
USER_AGENT = (
    "MorningEdge/0.1 (research project; "
    "+https://github.com/datanerdaditya/MorningEdge)"
)

# SEC requires explicit email identification per their fair-access policy.
# https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT = "MorningEdge research singhaditya7077@gmail.com"

REQUEST_TIMEOUT_SECONDS = 15.0

# Query params we strip during URL canonicalisation. Anything starting with
# 'utm_' is also stripped (handled in the function below).
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "ref_url",
    "_hsenc",
    "_hsmi",
    "yclid",
    "msclkid",
    "igshid",
}


# ---------------------------------------------------------------------------
# URL canonicalisation
# ---------------------------------------------------------------------------


def canonicalise_url(url: str) -> str:
    """Strip tracking parameters and normalise scheme/host casing.

    Same logical article must always produce the same canonical URL,
    regardless of which outlet's tracking decoration we received it with.
    """
    parsed = urlparse(url.strip())

    # Lower-case scheme and netloc; everything else preserved.
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()

    # Filter out known tracking params and anything starting with 'utm_'.
    cleaned_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS
    ]
    cleaned_query = urlencode(cleaned_params)

    # Drop fragments — they're never part of article identity.
    return urlunparse(
        (scheme, netloc, parsed.path.rstrip("/"), parsed.params, cleaned_query, "")
    )


# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _fetch_url(
    client: httpx.AsyncClient,
    url: str,
    extra_headers: dict | None = None,
) -> str:
    """Fetch a URL with retries. Raises on final failure."""
    response = await client.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=extra_headers or {},
    )
    response.raise_for_status()
    return response.text


async def _fetch_source(
    client: httpx.AsyncClient, source: Source
) -> list[RawArticle]:
    """Fetch one source. Returns [] on any failure (logged, not raised)."""
    # SEC requires a specific UA format with contact email
    extra_headers = {}
    if source.id.startswith("sec_"):
        extra_headers["User-Agent"] = SEC_USER_AGENT

    try:
        body = await _fetch_url(client, source.url, extra_headers=extra_headers)
    except Exception as e:
        logger.warning(f"[{source.id}] fetch failed: {type(e).__name__}: {e}")
        return []

    feed = feedparser.parse(body)
    if feed.bozo and not feed.entries:
        logger.warning(f"[{source.id}] feed parse failed: {feed.bozo_exception}")
        return []

    raws: list[RawArticle] = []
    for entry in feed.entries:
        try:
            raws.append(
                RawArticle(
                    title=entry.get("title", "").strip(),
                    url=entry.get("link", "").strip(),
                    source_id=source.id,
                    published_at=_parse_published(entry),
                    description=entry.get("summary", "")[:1000] or None,
                    raw_payload=dict(entry),
                )
            )
        except Exception as e:
            logger.debug(f"[{source.id}] skipped entry: {e}")

    logger.info(f"[{source.id}] fetched {len(raws)} entries")
    return raws


def _parse_published(entry: dict) -> datetime | None:
    """Best-effort parse of a feed entry's publication time."""
    # feedparser sometimes pre-parses to a struct_time
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    # Fall back to dateutil on the raw string
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            dt = dateparser.parse(raw)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass

    return None


# ---------------------------------------------------------------------------
# Validation: RawArticle -> Article
# ---------------------------------------------------------------------------


def _validate(raw: RawArticle, source: Source) -> Article | None:
    """Promote a RawArticle to a validated Article, or return None on failure."""
    if not raw.title or not raw.url:
        return None

    canonical = canonicalise_url(raw.url)
    if not canonical:
        return None

    try:
        return Article(
            article_id=make_article_id(canonical),
            title=raw.title,
            url=raw.url,
            canonical_url=canonical,
            source_id=raw.source_id,
            source_tier=source.tier,
            published_at=raw.published_at or datetime.now(timezone.utc),
            description=raw.description,
        )
    except Exception as e:
        logger.debug(f"[{raw.source_id}] validation failed for '{raw.title[:60]}': {e}")
        return None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def fetch_all_async(sources: list[Source] | None = None) -> list[Article]:
    """Fetch all sources concurrently. Returns validated Articles only."""
    sources = sources or all_sources()

    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml, */*"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_fetch_source(client, s) for s in sources),
            return_exceptions=False,
        )

    # Map source_id -> Source for tier lookups during validation
    source_map = {s.id: s for s in sources}

    articles: list[Article] = []
    for raws in results:
        for raw in raws:
            article = _validate(raw, source_map[raw.source_id])
            if article is not None:
                articles.append(article)

    logger.info(
        f"fetch_all complete: {len(articles)} articles from "
        f"{sum(1 for r in results if r)}/{len(sources)} sources"
    )
    return articles


def fetch_all(sources: list[Source] | None = None) -> list[Article]:
    """Synchronous wrapper around ``fetch_all_async`` for convenience."""
    return asyncio.run(fetch_all_async(sources))