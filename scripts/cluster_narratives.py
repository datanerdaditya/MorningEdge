"""Cluster recent enriched articles into narratives, summarise via Gemini.

Reads from DB, runs HDBSCAN per asset class, asks Gemini for theme labels,
persists results to the narratives table.

Usage:
    python scripts/cluster_narratives.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from morningedge.aggregation.clustering import cluster_all_classes
from morningedge.aggregation.themes import summarise_cluster
from morningedge.storage.db import (
    get_articles_for_clustering,
    write_cluster_assignments,
    write_narratives,
)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    articles = get_articles_for_clustering(days_back=2)
    logger.info(f"Loaded {len(articles)} articles for clustering")

    if not articles:
        logger.warning("No articles to cluster.")
        return

    # --- Step 1: bucket articles by asset class ---
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for art in articles:
        text = art["title"] + ". " + (art["description"] or "")
        for r in art["routings"]:
            by_class[r["asset_class"]].append((art["article_id"], text))

    # --- Step 2: cluster within each class ---
    assignments = cluster_all_classes(by_class)

    # --- Step 3: summarise each non-noise cluster ---
    # Group assignments back by cluster_id and gather their headlines
    titles_by_cluster: dict[str, list[str]] = defaultdict(list)
    cluster_to_class: dict[str, str] = {}
    article_titles = {a["article_id"]: a["title"] for a in articles}

    for asn in assignments:
        if "noise" in asn.cluster_id:
            continue
        titles_by_cluster[asn.cluster_id].append(article_titles[asn.article_id])
        cluster_to_class[asn.cluster_id] = asn.asset_class

    logger.info(f"Generating themes for {len(titles_by_cluster)} clusters")

    narratives = []
    for cluster_id, headlines in titles_by_cluster.items():
        # Skip 1-article clusters — they're not narratives
        if len(headlines) < 2:
            continue
        theme = summarise_cluster(headlines)
        if theme is None:
            logger.warning(f"No theme for {cluster_id}; skipping")
            continue
        narratives.append(
            {
                "narrative_id": f"{date.today().isoformat()}_{cluster_id}",
                "narrative_date": date.today(),
                "cluster_id": cluster_id,
                "asset_class": cluster_to_class[cluster_id],
                "title": theme.title,
                "summary": theme.summary,
                "article_count": len(headlines),
            }
        )

    n_clusters_written = write_cluster_assignments(assignments)
    n_narratives = write_narratives(narratives)

    logger.info(f"Wrote {n_clusters_written} cluster assignments and {n_narratives} narratives")

    # --- Print a summary ---
    print()
    print("=" * 90)
    print("  NARRATIVES")
    print("=" * 90)
    for n in sorted(narratives, key=lambda x: -x["article_count"]):
        print(f"\n  [{n['asset_class']}]  {n['title']}")
        print(f"  {n['article_count']} articles · {n['summary']}")


if __name__ == "__main__":
    main()