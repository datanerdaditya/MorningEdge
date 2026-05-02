"""Tests for the narrative clustering layer."""

import numpy as np
import pytest

from morningedge.aggregation.clustering import (
    ClusterAssignment,
    cluster_within_asset_class,
)


def test_empty_input():
    assignments = cluster_within_asset_class([], np.zeros((0, 768)), "rates")
    assert assignments == []


def test_too_few_articles_all_noise():
    """With only 1 article, can't form a cluster."""
    vecs = np.random.randn(1, 768).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    assignments = cluster_within_asset_class(["a1"], vecs, "rates")
    assert len(assignments) == 1
    assert "noise" in assignments[0].cluster_id


def test_clear_clusters_get_grouped():
    """Articles with very similar embeddings should land in the same cluster."""
    # Build 6 vectors: two tight clusters of 3 each
    rng = np.random.default_rng(42)
    base_a = rng.standard_normal(768).astype(np.float32)
    base_b = rng.standard_normal(768).astype(np.float32)

    def jitter(base, noise=0.05):
        v = base + noise * rng.standard_normal(768).astype(np.float32)
        return v / np.linalg.norm(v)

    cluster_a = np.stack([jitter(base_a) for _ in range(3)])
    cluster_b = np.stack([jitter(base_b) for _ in range(3)])
    all_vecs = np.vstack([cluster_a, cluster_b])
    ids = [f"a{i}" for i in range(6)]

    assignments = cluster_within_asset_class(ids, all_vecs, "rates")

    # First 3 should share a cluster, last 3 should share another
    cluster_ids = [a.cluster_id for a in assignments]
    assert cluster_ids[0] == cluster_ids[1] == cluster_ids[2]
    assert cluster_ids[3] == cluster_ids[4] == cluster_ids[5]
    assert cluster_ids[0] != cluster_ids[3]


def test_cluster_id_includes_asset_class():
    """Cluster IDs should be namespaced by asset class."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((4, 768)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    a = cluster_within_asset_class(["a1", "a2", "a3", "a4"], vecs, "rates")
    for asn in a:
        assert asn.cluster_id.startswith("rates_")