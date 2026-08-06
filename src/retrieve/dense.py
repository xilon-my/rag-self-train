"""Dense retrieval over precomputed embeddings. Vectors are L2-normalized, so
inner product == cosine. Pure numpy (no vector DB) — the ANN story is a served
detail; at ~1k-2k chunks exact top-k is instant and deterministic.
"""
from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, doc_ids: list[str], vectors: np.ndarray):
        self.doc_ids = doc_ids
        self.mat = vectors  # (N, dim), rows L2-normalized

    def search(self, qvec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        sims = self.mat @ qvec  # (N,)
        order = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]
