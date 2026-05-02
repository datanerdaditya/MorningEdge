"""Tests for entity extraction and event classification.

These load real models (GLiNER + BART-MNLI). First run downloads
~1.7GB total. Subsequent runs are cached.
"""

import pytest

from morningedge.enrichment.entities import Entity, extract_entities
from morningedge.enrichment.events import EVENT_TYPES, EventResult, classify_event


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_extract_entities_empty_input():
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_extract_entities_basic_company():
    """A clear company mention should be extracted."""
    text = "JPMorgan Chase reports record quarterly profit"
    entities = extract_entities(text)
    company_texts = [e.text for e in entities if e.label == "company"]
    assert any("JPMorgan" in t for t in company_texts), f"Got entities: {entities}"


def test_extract_entities_money():
    """Money amounts should be detected."""
    text = "Apollo launches new fund with $5 billion target"
    entities = extract_entities(text)
    has_money = any(e.label == "money_amount" for e in entities)
    assert has_money, f"Expected money entity, got {entities}"


def test_extract_entities_person():
    """Named individuals should be detected."""
    text = "Fed Chair Jerome Powell speaks at Jackson Hole symposium"
    entities = extract_entities(text)
    persons = [e.text for e in entities if e.label == "person"]
    assert any("Powell" in p for p in persons), f"Got: {entities}"


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------


def test_event_taxonomy_has_other_fallback():
    """The 'other' bucket must exist for low-confidence classifications."""
    assert "other" in EVENT_TYPES


def test_classify_event_empty_input():
    result = classify_event("")
    assert result.event_type == "other"


def test_classify_earnings_headline():
    """An earnings beat should classify as 'earnings'."""
    text = "JPMorgan beats Q3 estimates with record trading revenue"
    result = classify_event(text)
    assert result.event_type == "earnings", f"Got {result}"


def test_classify_central_bank_headline():
    """A Fed decision should classify as 'central_bank'."""
    text = "Federal Reserve cuts interest rates by 25 basis points"
    result = classify_event(text)
    assert result.event_type == "central_bank", f"Got {result}"


def test_classify_default_headline():
    """A default story should classify as 'default_distress'."""
    text = "Company files for Chapter 11 bankruptcy after defaulting on $2bn of debt"
    result = classify_event(text)
    assert result.event_type == "default_distress", f"Got {result}"


def test_classify_returns_score():
    """All classifications return a confidence score."""
    result = classify_event("Some news happened today")
    assert 0.0 <= result.score <= 1.0