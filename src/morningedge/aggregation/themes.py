"""Theme summarisation via Gemini Flash-Lite.

For each cluster, we ask Gemini to generate a short human-readable
theme name (3-7 words) and a one-line summary. We use the cheapest
Gemini tier (Flash-Lite, 1,000 RPD) since this runs on every cluster
in every pipeline run.

Single-article "clusters" (cluster_size == 1) skip Gemini entirely
and use the article's own title.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from morningedge.llm.gemini import MODEL_FLASH_LITE, call_gemini


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """A short human-readable name + summary for a cluster of articles."""

    title: str       # 3-7 words, e.g. "Fed pivot expectations rise"
    summary: str     # one sentence


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_THEME_PROMPT = """\
You are summarising a cluster of related news headlines for a credit-markets dashboard.

Headlines:
{headlines}

Return a JSON object with these fields:
- "title": a 3-7 word theme name (no quotes around it, just the words)
- "summary": one sentence (15-25 words) describing the underlying narrative

Keep it factual, concise, and finance-savvy. Avoid generic phrases like "company news"
or "market update". Be specific.

Examples:
  Title: "Fed pivot expectations strengthen"
  Summary: "Markets are pricing in earlier rate cuts as recent inflation prints come in below consensus."

  Title: "Private credit fund launches accelerate"
  Summary: "Apollo, Ares, and Blackstone announce major direct lending funds targeting middle-market borrowers."

Return ONLY the JSON object, no other text.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarise_cluster(headlines: list[str]) -> Theme | None:
    """Generate a theme name + summary for a list of headlines.

    Returns None on parse failure.
    """
    if not headlines:
        return None

    if len(headlines) == 1:
        # Single-article "cluster" — just use the headline itself
        return Theme(title=headlines[0][:80], summary=headlines[0])

    bulleted = "\n".join(f"- {h}" for h in headlines[:15])  # cap context
    prompt = _THEME_PROMPT.format(headlines=bulleted)

    try:
        raw = call_gemini(prompt, model=MODEL_FLASH_LITE, response_json=True)
    except Exception as e:
        logger.warning(f"Gemini theme call failed: {e}")
        return None

    try:
        data = json.loads(raw)
        title = str(data["title"]).strip().strip('"')
        summary = str(data["summary"]).strip()
        if not title or not summary:
            return None
        return Theme(title=title, summary=summary)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Theme parse failed: {e}; raw={raw[:200]}")
        return None