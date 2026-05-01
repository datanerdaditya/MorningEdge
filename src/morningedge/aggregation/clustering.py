"""Narrative clustering — find storylines, not just sentiment.

Uses HDBSCAN over sentence-transformer embeddings to group related stories
into themes. This is what lets us tell the user *why* sentiment is moving,
not just *that* it is.
"""
