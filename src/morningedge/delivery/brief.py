"""Daily morning brief generator.

Pulls today's aggregated state — regime, top narratives, notable entities,
bullish/bearish leaders — and asks Gemini Pro to write a one-page brief
in the voice of a buy-side credit analyst.

This is the only Gemini Pro call in the whole pipeline. One per day,
high stakes, worth the smarter model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from loguru import logger

from morningedge.dashboard.queries import (
    all_asset_class_summaries,
    latest_narratives,
    overall_summary,
    regime_label,
)
from morningedge.llm.gemini import MODEL_PRO, _bump_quota, _get_client, _wait_for_quota
from morningedge.storage.db import connect

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Brief:
    brief_date: date
    headline: str
    body_markdown: str
    regime: str
    avg_sentiment: float
    n_articles: int
    model: str = MODEL_PRO


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _gather_context(days_back: int = 1) -> dict:
    """Pull everything Gemini needs to write the brief."""
    overall = overall_summary(days_back=days_back)
    summaries = all_asset_class_summaries(days_back=days_back)

    # Top narratives by article count
    narratives_df = latest_narratives(limit=8)
    narratives = (
        narratives_df.to_dict(orient="records") if not narratives_df.empty else []
    )

    # Top bullish + bearish articles
    cutoff = datetime.now(UTC) - timedelta(days=days_back)
    with connect() as conn:
        bull_rows = conn.execute(
            """
            SELECT title, source_id, sentiment_score
            FROM articles
            WHERE sentiment_score IS NOT NULL
              AND published_at >= ?
            ORDER BY sentiment_score DESC
            LIMIT 5
            """,
            [cutoff],
        ).fetchall()
        bear_rows = conn.execute(
            """
            SELECT title, source_id, sentiment_score
            FROM articles
            WHERE sentiment_score IS NOT NULL
              AND published_at >= ?
            ORDER BY sentiment_score ASC
            LIMIT 5
            """,
            [cutoff],
        ).fetchall()

    return {
        "regime": regime_label(overall["avg_sentiment"]),
        "avg_sentiment": overall["avg_sentiment"],
        "n_articles": overall["n_articles"],
        "n_positive": overall["n_positive"],
        "n_negative": overall["n_negative"],
        "asset_classes": [
            {
                "id": s["asset_class_id"],
                "label": s["label"],
                "tier": s["tier"],
                "n": s["n_articles"],
                "sentiment": s["avg_sentiment"],
            }
            for s in summaries
            if s["n_articles"] > 0
        ],
        "narratives": [
            {
                "asset_class": n["asset_class"],
                "title": n["title"],
                "summary": n["summary"],
                "n_articles": n["article_count"],
            }
            for n in narratives
        ],
        "top_bullish": [
            {"title": t, "source": s, "score": float(sc)}
            for t, s, sc in bull_rows
        ],
        "top_bearish": [
            {"title": t, "source": s, "score": float(sc)}
            for t, s, sc in bear_rows
        ],
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_BRIEF_PROMPT = """\
You are the senior author of MorningEdge, a daily brief for a buy-side
credit analyst. The audience runs leveraged loans and private credit at a
mid-sized firm. They have ~90 seconds to read this. Write like an experienced
strategist, not a news aggregator.

Voice and rules:
- Tight, factual, professional. No fluff, no greetings, no "as an AI".
- Use the data provided below. Do not invent stories or names.
- If the data is thin (low article count, no clear narratives), SAY SO and
  keep the brief shorter. Don't pad.
- Markdown output. Stick to the exact structure below.
- Total length: under 350 words.

Required structure:

# Morning Brief — {today}

**One-liner:** <single sentence — the most important thing>

## The Tape
<one paragraph: regime call + the macro/credit context driving it>

## What Moved
<bullet list of the top 3-5 narratives, each one short. Format:
- **[asset_class]** Headline. One-clause why it matters.>

## Watchlist
<2-3 short bullets: themes/entities trending up that the analyst should track.
If the data doesn't support a watchlist, write "Nothing notable.">

## Bearish Risks
<2-3 bullets flagging the most negative stories. If sentiment is broadly
positive, this section may be very short or "Nothing material today.">

---

Data for today ({today}):

Regime: {regime} (avg sentiment {avg_sentiment:+.2f}, {n_articles} articles, {n_positive} pos / {n_negative} neg)

Per-asset-class sentiment (only classes with articles):
{asset_class_block}

Active narratives:
{narratives_block}

Top bullish today:
{bullish_block}

Top bearish today:
{bearish_block}
"""


def _format_context(ctx: dict) -> str:
    """Turn the gathered context dict into the prompt block."""
    asset_lines = "\n".join(
        f"  - [{a['tier']}] {a['label']:<25} {a['sentiment']:+.2f}  ({a['n']} articles)"
        for a in sorted(ctx["asset_classes"], key=lambda x: -x["n"])
    ) or "  (no data)"

    narr_lines = (
        "\n".join(
            f"  - [{n['asset_class']}] {n['title']} — {n['summary']} ({n['n_articles']} articles)"
            for n in ctx["narratives"]
        )
        or "  (no narratives)"
    )

    bull_lines = (
        "\n".join(
            f"  {b['score']:+.2f} [{b['source']}] {b['title']}"
            for b in ctx["top_bullish"]
        )
        or "  (none)"
    )

    bear_lines = (
        "\n".join(
            f"  {b['score']:+.2f} [{b['source']}] {b['title']}"
            for b in ctx["top_bearish"]
        )
        or "  (none)"
    )

    return _BRIEF_PROMPT.format(
        today=date.today().isoformat(),
        regime=ctx["regime"],
        avg_sentiment=ctx["avg_sentiment"],
        n_articles=ctx["n_articles"],
        n_positive=ctx["n_positive"],
        n_negative=ctx["n_negative"],
        asset_class_block=asset_lines,
        narratives_block=narr_lines,
        bullish_block=bull_lines,
        bearish_block=bear_lines,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_brief(days_back: int = 1, persist: bool = True) -> Brief:
    """Build today's brief. Tries Gemini Pro first, falls back to Flash if quota is hit."""
    ctx = _gather_context(days_back=days_back)
    prompt = _format_context(ctx)

    logger.info(f"Brief context: {ctx['n_articles']} articles, regime={ctx['regime']}")

    body, model_used = _generate_with_fallback(prompt)

    # Pull the one-liner out of the body for storage / preview
    headline = _extract_headline(body)

    brief = Brief(
        brief_date=date.today(),
        headline=headline,
        body_markdown=body,
        regime=ctx["regime"],
        avg_sentiment=ctx["avg_sentiment"],
        n_articles=ctx["n_articles"],
        model=model_used,
    )

    if persist:
        _persist_brief(brief)

    return brief


def get_latest_brief() -> Brief | None:
    """Return today's brief if it exists, else the most recent."""
    with connect() as conn:
        row = conn.execute(
            """
            SELECT brief_date, headline, body_markdown, regime,
                   avg_sentiment, n_articles, model
            FROM briefs
            ORDER BY brief_date DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    return Brief(
        brief_date=row[0],
        headline=row[1],
        body_markdown=row[2],
        regime=row[3] or "Mixed",
        avg_sentiment=float(row[4] or 0.0),
        n_articles=int(row[5] or 0),
        model=row[6] or MODEL_PRO,
    )

def _generate_with_fallback(prompt: str) -> tuple[str, str]:
    """Try Pro, then Flash, then Flash-Lite. Return (body, model_used)."""
    from morningedge.llm.gemini import MODEL_FLASH, MODEL_FLASH_LITE

    client = _get_client()
    candidates = [MODEL_PRO, MODEL_FLASH, MODEL_FLASH_LITE]

    last_error: Exception | None = None
    for model in candidates:
        try:
            _wait_for_quota(model)
            logger.info(f"Brief: trying {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            _bump_quota(model)
            body = (response.text or "").strip()
            if body:
                logger.info(f"Brief: succeeded with {model}")
                return body, model
        except Exception as e:
            last_error = e
            err_text = str(e)
            # 429 is quota; 500-class errors might transiently work later but
            # for the brief use case we just keep trying smaller models.
            if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                logger.warning(f"{model} quota exhausted; falling back")
                continue
            logger.warning(f"{model} failed: {type(e).__name__}; falling back")
            continue

    raise RuntimeError(
        f"All Gemini models failed for brief generation. Last error: {last_error}"
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_headline(body: str) -> str:
    """Pull the one-liner from the Markdown body. Falls back to first line."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("**one-liner:**"):
            # Strip the prefix and any leading/trailing markdown
            return stripped.split(":", 1)[1].strip().strip("*").strip()
    # Fallback: first non-empty, non-header line
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:200]
    return "Daily brief generated"


def _persist_brief(brief: Brief) -> None:
    now = datetime.now(UTC)
    with connect() as conn:
        # Idempotent re-runs: replace today's brief
        conn.execute("DELETE FROM briefs WHERE brief_date = ?", [brief.brief_date])
        conn.execute(
            """
            INSERT INTO briefs (
                brief_date, headline, body_markdown, regime,
                avg_sentiment, n_articles, computed_at, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                brief.brief_date,
                brief.headline,
                brief.body_markdown,
                brief.regime,
                brief.avg_sentiment,
                brief.n_articles,
                now,
                brief.model,
            ],
        )