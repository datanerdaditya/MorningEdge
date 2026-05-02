"""Narrative clustering with HDBSCAN.

We reuse the mpnet embeddings already computed for fuzzy dedup and
asset-class routing — no new model required. HDBSCAN groups dense
neighborhoods of articles into "clusters" (= narratives), and tags
articles too far from any neighborhood as "noise".

We cluster *per asset class*, not globally. A rates story and a tech
story should never share a cluster, even if their embeddings overlap
slightly. Per-class clustering also gives finer narratives.

Output
------
``cluster_articles`` returns a dict: asset_class -> [(cluster_id, [article_ids])].
Cluster IDs are local to the asset class — "rates_0", "rates_1", "tech_0"
— so the same article in two asset classes can sit in different clusters.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
from loguru import logger

from morningedge.ingestion.dedup import embed_texts

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Articles in fewer than this many neighbours don't form a cluster.
# 2 is the smallest meaningful narrative ("at least two outlets covered this").
MIN_CLUSTER_SIZE = 2

# Within-cluster connectivity. Lower = more, smaller clusters.
# 1 is a sensible default for short-text news clustering.
MIN_SAMPLES = 1

# Cosine distance threshold for joining a cluster. We let HDBSCAN decide
# but cap how loose clusters can get.
CLUSTER_SELECTION_EPSILON = 0.4


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterAssignment:
    """One article's membership in a cluster within an asset class."""

    article_id: str
    asset_class: str
    cluster_id: str   # e.g. "rates_3", or "rates_noise"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_within_asset_class(
    article_ids: list[str],
    embeddings: np.ndarray,
    asset_class: str,
) -> list[ClusterAssignment]:
    """Cluster a set of articles within a single asset class.

    Parameters
    ----------
    article_ids : list of str
    embeddings  : (n, d) numpy array, one row per article
    asset_class : the asset_class id this batch belongs to

    Returns
    -------
    list of ClusterAssignment, one per input article. Articles HDBSCAN
    can't assign get cluster_id = "{asset_class}_noise".
    """
    if len(article_ids) == 0:
        return []

    if len(article_ids) < MIN_CLUSTER_SIZE:
        # Too few to cluster — everyone is noise
        return [
            ClusterAssignment(
                article_id=aid,
                asset_class=asset_class,
                cluster_id=f"{asset_class}_noise",
            )
            for aid in article_ids
        ]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        metric="euclidean",  # mpnet is L2-normalised so euclidean ~= cosine
    )

    # Convert L2-normalised vectors to cosine distance via euclidean
    labels = clusterer.fit_predict(embeddings)

    assignments: list[ClusterAssignment] = []
    for aid, label in zip(article_ids, labels, strict=False):
        cluster_id = (
            f"{asset_class}_noise" if label == -1 else f"{asset_class}_{label}"
        )
        assignments.append(
            ClusterAssignment(
                article_id=aid, asset_class=asset_class, cluster_id=cluster_id
            )
        )

    n_clusters = len(set(a.cluster_id for a in assignments if "noise" not in a.cluster_id))
    n_noise = sum(1 for a in assignments if "noise" in a.cluster_id)
    logger.info(
        f"[cluster] {asset_class}: {len(article_ids)} articles -> "
        f"{n_clusters} clusters + {n_noise} noise"
    )

    return assignments


def cluster_all_classes(
    articles_by_class: dict[str, list[tuple[str, str]]],
) -> list[ClusterAssignment]:
    """Cluster every asset class in one call.

    Parameters
    ----------
    articles_by_class : dict mapping asset_class -> list of (article_id, text).
                        ``text`` should be title + ". " + description.

    Returns
    -------
    Flat list of ClusterAssignments across all classes.
    """
    all_assignments: list[ClusterAssignment] = []

    for asset_class, items in articles_by_class.items():
        if not items:
            continue
        ids = [aid for aid, _ in items]
        texts = [txt for _, txt in items]
        vecs = embed_texts(texts)
        all_assignments.extend(
            cluster_within_asset_class(ids, vecs, asset_class)
        )

    return all_assignments