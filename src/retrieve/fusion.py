"""RRF (Reciprocal Rank Fusion) — from first principles, the five-line formula.

score(doc) = Σ 1 / (k + rank(doc)) over the merged lists. Rank-based, so it is
scale-invariant across BM25 and dense score distributions.
"""
from __future__ import annotations

from collections import defaultdict


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: one ordered doc-id list per retriever. Returns (doc_id, score) desc."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
