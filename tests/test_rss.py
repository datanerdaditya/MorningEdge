"""Tests for the RSS ingestion layer."""

from morningedge.ingestion.rss import canonicalise_url


def test_canonicalise_strips_utm():
    url = "https://example.com/article?utm_source=twitter&utm_campaign=x"
    assert canonicalise_url(url) == "https://example.com/article"


def test_canonicalise_strips_known_trackers():
    url = "https://example.com/article?fbclid=abc123&gclid=xyz"
    assert canonicalise_url(url) == "https://example.com/article"


def test_canonicalise_preserves_real_query_params():
    url = "https://example.com/search?q=fed+rates&page=2"
    result = canonicalise_url(url)
    assert "q=fed" in result
    assert "page=2" in result


def test_canonicalise_drops_fragment():
    url = "https://example.com/article#section-3"
    assert canonicalise_url(url) == "https://example.com/article"


def test_canonicalise_lowers_scheme_and_host():
    url = "HTTPS://EXAMPLE.COM/Article"
    assert canonicalise_url(url) == "https://example.com/Article"


def test_canonicalise_strips_trailing_slash():
    url = "https://example.com/article/"
    assert canonicalise_url(url) == "https://example.com/article"