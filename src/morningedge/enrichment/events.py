"""Event classification via zero-shot NLI.

Uses BART-MNLI (or DeBERTa-v3) to classify each article into event types
without training data: earnings, M&A, regulatory, central-bank action,
default/distress, fundraising, ratings change, executive change, macro print.
"""
