"""Gemini client with model routing.

The free tier has different daily quotas per model:
    Flash-Lite : 1,000 RPD — bulk summarisation, theme labels
    Flash      :   250 RPD — top-story sentiment re-scoring
    Pro        :   100 RPD — daily brief, complex chat queries

This module is the single chokepoint for all Gemini calls. It tracks
quota usage, applies the right model per task type, and falls back
gracefully when limits are hit.
"""
