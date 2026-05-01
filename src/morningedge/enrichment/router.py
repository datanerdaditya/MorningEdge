"""Asset-class routing via semantic similarity.

Replaces brittle keyword queries with embedding-based routing: each article
is matched against the asset-class descriptions in ``taxonomy.py`` using
sentence-transformer cosine similarity. Multi-label: an article can belong
to several classes (e.g. a Fed decision is both 'rates' and 'us_macro').
"""
