"""DuckDB schema definitions.

Tables (created on first run):
    articles      : raw normalised articles
    enrichments   : entities, events, sentiment per article
    routings      : article ↔ asset_class many-to-many
    daily_scores  : per (date, asset_class) aggregate
    narratives    : daily clusters with theme summaries
    embeddings    : article embeddings (BLOB, for fuzzy dedup + clustering)
"""
