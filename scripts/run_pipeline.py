"""End-to-end pipeline runner.

Invoked daily by GitHub Actions. Steps:
    1. Ingest (RSS + APIs)
    2. Dedup
    3. Enrich (sentiment, entities, events, routing)
    4. Aggregate (score, cluster, summarise themes)
    5. Generate daily brief
    6. Persist everything to DuckDB
"""

from loguru import logger


def main() -> None:
    logger.info("MorningEdge pipeline — placeholder. Wired up in Week 1.")


if __name__ == "__main__":
    main()
