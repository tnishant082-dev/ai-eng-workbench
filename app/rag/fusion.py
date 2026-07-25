"""Hybrid fusion: weighted blend + Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """RRF over lists of doc/chunk ids. Returns (id, score) sorted desc."""
    scores: dict[str, float] = defaultdict(float)
    weights = weights or [1.0] * len(ranked_lists)
    for w, ranking in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_hits(
    bm25_ids: list[str],
    dense_ids: list[str],
    id_to_payload: dict[str, Any],
    k: int = 60,
    alpha_bm25: float = 1.0,
    alpha_dense: float = 1.0,
) -> list[tuple[str, float]]:
    """RRF fuse BM25 and dense rankings; payload lookup left to caller."""
    return reciprocal_rank_fusion(
        [bm25_ids, dense_ids],
        k=k,
        weights=[alpha_bm25, alpha_dense],
    )
