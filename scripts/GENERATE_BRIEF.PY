"""Generate today's morning brief and print it to stdout.

Usage:
    python scripts/generate_brief.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.delivery.brief import generate_brief
from morningedge.storage.db import init_schema


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    # Make sure the briefs table exists if this is the first run after schema change
    init_schema()

    brief = generate_brief(days_back=2)

    print()
    print("=" * 80)
    print(f"  HEADLINE  {brief.headline}")
    print(f"  REGIME    {brief.regime}  ({brief.avg_sentiment:+.2f})")
    print(f"  ARTICLES  {brief.n_articles}")
    print("=" * 80)
    print()
    print(brief.body_markdown)


if __name__ == "__main__":
    main()