"""Smoke tests for the taxonomy module."""

from morningedge.taxonomy import TAXONOMY, by_id, by_tier


def test_taxonomy_non_empty():
    assert len(TAXONOMY) > 0


def test_hero_tier_has_credit_classes():
    hero_ids = {ac.id for ac in by_tier("hero")}
    assert {"lev_loans", "private_credit", "clo", "high_yield"} <= hero_ids


def test_by_id_returns_correct_class():
    ac = by_id("private_credit")
    assert ac is not None
    assert ac.tier == "hero"


def test_by_id_unknown_returns_none():
    assert by_id("does_not_exist") is None
