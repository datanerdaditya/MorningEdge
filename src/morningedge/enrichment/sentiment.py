"""Sentiment scoring — FinBERT for bulk, Gemini for top stories.

FinBERT (ProsusAI/finbert) handles every article cheaply and locally.
The N highest-impact stories per asset class get a second pass with Gemini
to capture nuance FinBERT misses (irony, conditional language, market
expectations vs. realised events).
"""
