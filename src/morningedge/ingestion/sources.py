"""Free news source registry for MorningEdge.

The curated list below is the heart of the ingestion layer. Sources are
chosen for relevance to leveraged finance + private credit, with macro
context layered on. We deliberately favour fewer, higher-quality sources
over a long tail of aggregators — the noise floor in finance news is
already brutal.

Each source has:
    - id            : stable identifier used everywhere downstream
    - name          : human-readable label
    - url           : the RSS feed URL
    - tier          : quality tier (Tier 1 outlets > Tier 2 specialists > Tier 3)
    - default_assets: which asset classes this source primarily covers
                      (used as a prior; the embedding router can override)

Adding a source
---------------
Test it first:
    >>> import feedparser
    >>> feed = feedparser.parse("https://example.com/feed")
    >>> len(feed.entries) > 0  # should return True
    >>> feed.entries[0].title  # should look like a headline
"""

from __future__ import annotations

from dataclasses import dataclass

from morningedge.ingestion.models import SourceTier


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    url: str
    tier: SourceTier
    default_assets: list[str]  # asset_class ids from taxonomy.py


# ---------------------------------------------------------------------------
# Curated source list
# ---------------------------------------------------------------------------
# Notes on inclusion/exclusion:
#   - Reuters has the cleanest credit-relevant wire feed of the free outlets.
#   - Federal Reserve / ECB feeds are gold for rates context.
#   - We'd love S&P LCD, PitchBook, and Private Debt Investor — most are
#     paywalled or don't expose RSS. We work with what's free.
#   - SEC EDGAR full-text RSS is free and structured; we use it selectively
#     for 8-K / 10-K filings tagged with credit-relevant items.
# ---------------------------------------------------------------------------

SOURCES: list[Source] = [
    # --- Tier 1: Wire-style coverage ---
    Source(
        id="investing_news",
        name="Investing.com — News",
        url="https://www.investing.com/rss/news.rss",
        tier=SourceTier.TIER_1,
        default_assets=["us_macro", "rates", "risk_equity"],
    ),
    Source(
        id="marketwatch_top",
        name="MarketWatch — Top Stories",
        url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
        tier=SourceTier.TIER_1,
        default_assets=["us_macro", "risk_equity", "banks"],
    ),
    Source(
        id="ft_alphaville",
        name="FT Alphaville",
        url="https://www.ft.com/alphaville?format=rss",
        tier=SourceTier.TIER_1,
        default_assets=["high_yield", "lev_loans", "rates", "private_credit"],
    ),
    # --- Tier 1: Central banks (raw policy primary source) ---
    Source(
        id="fed_press",
        name="Federal Reserve — Press Releases",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        tier=SourceTier.TIER_1,
        default_assets=["rates", "us_macro", "banks"],
    ),
    Source(
        id="ecb_press",
        name="ECB — Press Releases",
        url="https://www.ecb.europa.eu/rss/press.html",
        tier=SourceTier.TIER_1,
        default_assets=["rates", "europe_macro"],
    ),
    Source(
        id="boe_news",
        name="Bank of England — News",
        url="https://www.bankofengland.co.uk/rss/news",
        tier=SourceTier.TIER_1,
        default_assets=["rates", "europe_macro"],
    ),
    Source(
        id="fred_blog",
        name="FRED Blog (St. Louis Fed)",
        url="https://fredblog.stlouisfed.org/feed/",
        tier=SourceTier.TIER_1,
        default_assets=["us_macro", "rates"],
    ),
    # --- Tier 2: Specialist / sector-relevant ---
    Source(
        id="sec_8k",
        name="SEC EDGAR — Recent 8-K Filings",
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=include&count=40&output=atom",
        tier=SourceTier.TIER_2,
        default_assets=["banks", "tech", "energy", "healthcare"],
    ),
    # --- Tier 3: Aggregators ---
    Source(
        id="seeking_alpha_credit",
        name="Seeking Alpha — Bonds & Fixed Income",
        url="https://seekingalpha.com/feed.xml",
        tier=SourceTier.TIER_3,
        default_assets=["high_yield", "rates"],
    ),
    Source(
        id="yahoo_finance_top",
        name="Yahoo Finance — Top Stories",
        url="https://finance.yahoo.com/news/rssindex",
        tier=SourceTier.TIER_3,
        default_assets=["risk_equity", "us_macro"],
    ),
]


def by_id(source_id: str) -> Source | None:
    """Look up a source by its stable id."""
    return next((s for s in SOURCES if s.id == source_id), None)


def by_tier(tier: SourceTier) -> list[Source]:
    """Return all sources in a given quality tier."""
    return [s for s in SOURCES if s.tier == tier]


def all_sources() -> list[Source]:
    """Return the full source list (immutable copy)."""
    return list(SOURCES)