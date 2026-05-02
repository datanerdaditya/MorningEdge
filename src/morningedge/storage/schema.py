"""DuckDB schema definitions.

One module = one source of truth. If you want to know what tables and
columns exist in the MorningEdge database, look here. If you want to
change the schema, change it here, then add a migration if needed.

We're deliberately ahead of ourselves on the schema: columns for
Week 2 (sentiment, entities, events) and Week 3 (cluster_id, theme_id)
are pre-declared as NULLable. This avoids painful migrations during
the build.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------
#
# Run order: articles -> enrichments -> routings -> daily_scores -> narratives
#
# All tables use ``article_id`` (or ``narrative_id``) as the primary key
# where applicable. ``article_id`` is the deterministic hash from
# ``ingestion.models.make_article_id`` — same canonical URL always maps
# to the same id, across runs and machines.
# ---------------------------------------------------------------------------


ARTICLES_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id      VARCHAR PRIMARY KEY,
    title           VARCHAR NOT NULL,
    url             VARCHAR NOT NULL,
    canonical_url   VARCHAR NOT NULL,
    source_id       VARCHAR NOT NULL,
    source_tier     VARCHAR NOT NULL,
    description     VARCHAR,
    published_at    TIMESTAMP NOT NULL,
    fetched_at      TIMESTAMP NOT NULL,

    -- Populated in Week 2 by enrichment/sentiment.py
    sentiment_score DOUBLE,         -- -1.0 to 1.0 (FinBERT or Gemini)
    sentiment_label VARCHAR,        -- 'positive' | 'neutral' | 'negative'

    -- Populated in Week 2 by enrichment/events.py
    event_type      VARCHAR,        -- 'earnings' | 'm_and_a' | 'rates' | ...

    -- Populated in Week 2 by enrichment/entities.py (JSON array)
    entities        JSON,           -- [{"text": "JPMorgan", "type": "ORG"}, ...]

    -- Populated in Week 3 by aggregation/clustering.py
    cluster_id      VARCHAR,        -- HDBSCAN cluster, or NULL if noise

    -- Populated in Week 2 by enrichment/router.py (cached for performance)
    embedding       JSON,           -- sentence-transformer vector as JSON array

    -- Populated in Week 2 by enrichment pipeline (NULL = not yet enriched)
    enriched_at     TIMESTAMP
);
"""

ROUTINGS_DDL = """
CREATE TABLE IF NOT EXISTS routings (
    article_id      VARCHAR NOT NULL,
    asset_class     VARCHAR NOT NULL,
    score           DOUBLE NOT NULL,    -- cosine similarity, 0 to 1
    PRIMARY KEY (article_id, asset_class)
);
"""

DAILY_SCORES_DDL = """
CREATE TABLE IF NOT EXISTS daily_scores (
    score_date      DATE NOT NULL,
    asset_class     VARCHAR NOT NULL,
    sentiment       DOUBLE NOT NULL,        -- weighted avg, -1 to 1
    article_count   INTEGER NOT NULL,
    summary         VARCHAR,                -- one-line LLM summary
    computed_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (score_date, asset_class)
);
"""

NARRATIVES_DDL = """
CREATE TABLE IF NOT EXISTS narratives (
    narrative_id    VARCHAR PRIMARY KEY,    -- date + cluster_id
    narrative_date  DATE NOT NULL,
    cluster_id      VARCHAR NOT NULL,
    asset_class     VARCHAR,                -- the dominant class for this cluster
    title           VARCHAR NOT NULL,       -- short LLM-generated theme name
    summary         VARCHAR NOT NULL,       -- 1-2 sentence theme summary
    article_count   INTEGER NOT NULL,
    computed_at     TIMESTAMP NOT NULL
);
"""

# Useful indexes for common query patterns
INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_articles_cluster   ON articles(cluster_id);",
    "CREATE INDEX IF NOT EXISTS idx_routings_asset     ON routings(asset_class);",
    "CREATE INDEX IF NOT EXISTS idx_scores_date        ON daily_scores(score_date);",
]


# Public list — used by ``db.init_schema()`` to create everything in order.
ALL_DDL: list[str] = [
    ARTICLES_DDL,
    ROUTINGS_DDL,
    DAILY_SCORES_DDL,
    NARRATIVES_DDL,
    *INDEXES_DDL,
]