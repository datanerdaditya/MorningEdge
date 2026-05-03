"""End-to-end MorningEdge pipeline runner.

Steps:
    1. Init DB schema (idempotent).
    2. Fetch articles from all RSS sources.
    3. Fuzzy/semantic dedup.
    4. Persist new articles.
    5. Enrich unenriched articles:
       - FinBERT sentiment
       - Asset-class routing
       - Entity extraction
       - Event classification
       - Gemini top-story rescore for the most-important N
    6. Persist enrichments.
    7. (Week 3+) Cluster, generate brief.
"""

# TODO(Week 4): persist article embeddings during enrichment pass.
# Currently embeddings only get written during fuzzy dedup compare-step,
# leaving most articles with NULL embedding. Fix: in the enrich block,
# call embed_texts() and persist to articles.embedding.

from __future__ import annotations

from morningedge.ingestion.dedup import embed_texts
from morningedge.storage.db import persist_embeddings

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datetime import datetime, timezone

from loguru import logger

from morningedge.enrichment.entities import extract_entities_batch
from morningedge.enrichment.events import classify_events_batch
from morningedge.enrichment.router import route_texts
from morningedge.enrichment.sentiment import score_articles_batch
from morningedge.ingestion.dedup import fuzzy_dedupe
from morningedge.ingestion.rss import fetch_all
from morningedge.llm.gemini import quota_summary, rescore_sentiment
from morningedge.storage.db import (
    count_articles,
    count_enriched,
    get_unenriched_articles,
    init_schema,
    insert_articles,
    write_enrichments,
    write_routings,
)


# How many top-importance articles get Gemini-rescored per run.
# Stays well under Flash's 250/day quota.
TOP_RESCORE_N = 8


def main() -> int:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | {level: <7} | {message}",
    )

    started = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {started.isoformat()}")

    # --- Step 1: schema ---
    init_schema()

    # --- Step 2: fetch ---
    before_total = count_articles()
    articles = fetch_all()
    if not articles:
        logger.warning("No articles fetched. Skipping enrichment too.")
        return 1

    # --- Step 3: fuzzy dedup ---
    deduped = fuzzy_dedupe(articles)

    # --- Step 4: persist new articles ---
    inserted, skipped = insert_articles(deduped)
    after_total = count_articles()

    # --- Step 5: enrich any unenriched articles ---
    pending = get_unenriched_articles(limit=500)
    logger.info(f"Enriching {len(pending)} articles")

    if pending:
        titles = [a["title"] for a in pending]
        descriptions = [a["description"] for a in pending]
        full_texts = [
            f"{t}. {d}" if d else t for t, d in zip(titles, descriptions)
        ]

        # Run each model in batch
        sentiments = score_articles_batch(list(zip(titles, descriptions)))
        routings_per = route_texts(full_texts)
        entities_per = extract_entities_batch(full_texts)
        events_per = classify_events_batch(full_texts)

        # Persist embeddings — needed for the chat RAG layer
        embeddings = embed_texts(full_texts)
        article_ids_for_emb = [a["article_id"] for a in pending]
        persist_embeddings(article_ids_for_emb, embeddings)
        

        # --- Step 5b: pick top-N for Gemini rescore ---
        importance = []
        for i, (article, routings) in enumerate(zip(pending, routings_per)):
            top_score = max((r.score for r in routings), default=0.0)
            tier_boost = {"tier_1": 0.10, "tier_2": 0.05, "tier_3": 0.0}.get(
                article["source_tier"], 0.0
            )
            importance.append((top_score + tier_boost, i))

        importance.sort(reverse=True)
        rescore_indices = {i for _, i in importance[:TOP_RESCORE_N]}

        gemini_overrides = 0
        gemini_errors = 0
        for i in rescore_indices:
            article = pending[i]
            try:
                res = rescore_sentiment(article["title"], article["description"])
            except Exception as e:
                logger.warning(f"Gemini rescore error: {e}; falling back to FinBERT")
                gemini_errors += 1
                continue

            if res is not None:
                # Replace FinBERT's call with Gemini's
                from morningedge.enrichment.sentiment import SentimentResult
                sentiments[i] = SentimentResult(
                    score=res.score,
                    label=res.label,
                    p_positive=max(res.score, 0),
                    p_neutral=1 - abs(res.score),
                    p_negative=max(-res.score, 0),
                )
                gemini_overrides += 1

        logger.info(
            f"Gemini rescored {gemini_overrides}/{len(rescore_indices)} top articles"
            f" ({gemini_errors} errors)"
        )

        logger.info(f"Gemini rescored {gemini_overrides}/{len(rescore_indices)} top articles")

        # --- Step 6: build enrichment payloads & persist ---
        enrichment_rows = []
        routings_by_article: dict[str, list[dict]] = {}

        for article, sent, routings, entities, event in zip(
            pending, sentiments, routings_per, entities_per, events_per
        ):
            enrichment_rows.append(
                {
                    "article_id": article["article_id"],
                    "sentiment_score": sent.score,
                    "sentiment_label": sent.label,
                    "event_type": event.event_type,
                    "entities": [e.to_dict() for e in entities],
                }
            )
            if routings:
                routings_by_article[article["article_id"]] = [
                    {"asset_class_id": r.asset_class_id, "score": r.score}
                    for r in routings
                ]

        write_enrichments(enrichment_rows)
        n_routings = write_routings(routings_by_article)
        logger.info(f"Persisted: {len(enrichment_rows)} enrichments, {n_routings} routings")

    enriched, total = count_enriched()
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    print()
    print("=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Fetched          {len(articles):>5}")
    print(f"  After fuzzy      {len(deduped):>5}  (-{len(articles) - len(deduped)})")
    print(f"  New articles     {inserted:>5}  (skipped {skipped})")
    print(f"  Total in DB      {total:>5}  (was {before_total})")
    print(f"  Enriched         {enriched:>5} / {total}")
    print(f"  Gemini quota     {quota_summary()}")
    print(f"  Duration         {duration:>5.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())