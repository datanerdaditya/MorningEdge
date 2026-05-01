"""RSS feed ingestion (feedparser-based).

Pulls from the curated free-source list in ``sources.py``. Output: a list of
raw ``Article`` objects to hand off to the dedup layer.
"""
