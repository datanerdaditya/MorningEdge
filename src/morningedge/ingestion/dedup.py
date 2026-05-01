"""Article deduplication — exact and fuzzy.

- Exact: hash on (normalised title + url).
- Fuzzy: cosine similarity on sentence-transformer embeddings (catches
  syndicated stories where outlet A and outlet B publish the same Reuters
  wire copy with different titles).
"""
