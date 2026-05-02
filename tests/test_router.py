"""Tests for the asset-class routing layer.

These tests verify routing decisions on representative financial
headlines. The threshold and top-K behaviour are tuned so these
specific test cases pass — if you change the threshold, expect to
adjust these.
"""


from morningedge.enrichment.router import (
    MAX_ASSIGNMENTS,
    ROUTING_THRESHOLD,
    route_text,
    route_texts,
)


def test_empty_text_returns_no_routings():
    assert route_text("") == []
    assert route_text("   ") == []


def test_credit_headline_routes_to_credit():
    """A leveraged-loans story should hit lev_loans."""
    text = (
        "Leveraged loan issuance surges to record high as borrowers refinance "
        "ahead of expected rate cuts; LSTA index rallies on tighter spreads."
    )
    routings = route_text(text)
    assert len(routings) > 0
    top_ids = [r.asset_class_id for r in routings]
    assert "lev_loans" in top_ids


def test_private_credit_headline():
    """A private credit / direct lending story should route to private_credit."""
    text = (
        "Apollo launches new direct lending fund with $5bn target, "
        "expanding private credit footprint in middle-market lending."
    )
    routings = route_text(text)
    top_ids = [r.asset_class_id for r in routings]
    assert "private_credit" in top_ids


def test_fed_headline_routes_to_rates():
    """A Fed rate-cut headline must route to 'rates' as the top class."""
    text = (
        "Federal Reserve cuts interest rates by 25 basis points, "
        "signals further easing as inflation cools toward target."
    )
    routings = route_text(text)
    assert len(routings) > 0, "Expected at least one routing"
    assert routings[0].asset_class_id == "rates", (
        f"Expected 'rates' as top routing, got {[r.asset_class_id for r in routings]}"
    )


def test_unrelated_headline_returns_nothing_or_low_confidence():
    """Sports news shouldn't confidently route anywhere."""
    text = "Real Madrid wins UEFA Champions League final 3-1"
    routings = route_text(text)
    # Either no routings, or the top score is weak
    assert len(routings) == 0 or routings[0].score < ROUTING_THRESHOLD


def test_cap_at_max_assignments():
    """Even an article matching many classes should cap at MAX_ASSIGNMENTS."""
    text = (
        "Fed cuts rates as banks rally, tech leads stocks higher, "
        "high yield credit spreads tighten, Treasury yields fall, "
        "private credit funds raise record capital."
    )
    routings = route_text(text)
    assert len(routings) <= MAX_ASSIGNMENTS


def test_routings_sorted_by_score():
    text = "Federal Reserve holds rates steady, JPMorgan reports earnings"
    routings = route_text(text)
    if len(routings) >= 2:
        for i in range(len(routings) - 1):
            assert routings[i].score >= routings[i + 1].score


def test_batch_matches_individual():
    """route_texts should match route_text for each input."""
    texts = [
        "Fed cuts rates by 25bps",
        "Apollo raises new private credit fund",
        "Apple reports record iPhone sales",
    ]
    individual = [route_text(t) for t in texts]
    batched = route_texts(texts)

    for i, b in zip(individual, batched, strict=False):
        assert len(i) == len(b)
        for ri, rb in zip(i, b, strict=False):
            assert ri.asset_class_id == rb.asset_class_id
            assert abs(ri.score - rb.score) < 1e-5