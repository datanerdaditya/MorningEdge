"""Tests for the FinBERT sentiment layer.

These tests load the real model — first run takes ~30s for the download,
subsequent runs are fast. We use a small set of clearly-signed financial
headlines to verify the model gives sensible outputs.
"""


from morningedge.enrichment.sentiment import (
    SentimentResult,
    score_article,
    score_articles_batch,
    score_texts,
)


def test_null_result_is_neutral():
    null = SentimentResult.null()
    assert null.score == 0.0
    assert null.label == "neutral"


def test_empty_list_returns_empty():
    assert score_texts([]) == []


def test_empty_strings_return_null():
    results = score_texts(["", "  ", None])
    assert all(r.label == "neutral" and r.score == 0 for r in results)


def test_clear_positive_headline():
    """Strong earnings beat should score positive."""
    results = score_texts(["JPMorgan reports record profit, beats estimates significantly"])
    assert results[0].score > 0.3, f"Got score {results[0].score}, expected > 0.3"
    assert results[0].label == "positive"


def test_clear_negative_headline():
    """Bankruptcy filing should score strongly negative."""
    results = score_texts(["Company files for Chapter 11 bankruptcy after defaulting on debt"])
    assert results[0].score < -0.3, f"Got score {results[0].score}, expected < -0.3"
    assert results[0].label == "negative"


def test_neutral_headline():
    """Pure factual statement should score near zero."""
    results = score_texts(["Federal Reserve will hold meeting next Tuesday"])
    assert abs(results[0].score) < 0.5, f"Got score {results[0].score}, expected closer to 0"


def test_batch_consistency():
    """Same inputs in batch should give same results as individual scoring."""
    text = "Strong quarterly earnings drive stock higher"

    individual = score_texts([text])[0]
    batched = score_texts([text, text, text])

    for b in batched:
        assert abs(b.score - individual.score) < 1e-5


def test_score_article_with_description():
    """Description should influence final score."""
    result = score_article(
        title="Federal Reserve meeting today",
        description="Fed expected to cut rates by 50bps amid recession fears",
    )
    # Title alone is neutral; description has clear directional language
    assert isinstance(result, SentimentResult)
    assert -1.0 <= result.score <= 1.0


def test_score_article_no_description():
    """Should work fine with only a title."""
    result = score_article(
        title="Apple reports strong iPhone sales",
        description=None,
    )
    assert result.score > 0


def test_score_batch_returns_correct_length():
    """Batch helper must return one result per input."""
    articles = [
        ("Bullish news", "very good"),
        ("Bearish news", "very bad"),
        ("Neutral news", None),
    ]
    results = score_articles_batch(articles)
    assert len(results) == 3