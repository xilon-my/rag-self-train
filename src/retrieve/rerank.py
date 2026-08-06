"""Cross-encoder reranker (bge-reranker family). Scores (query, doc) pairs joined
into one sequence; run over the fused top-N candidates, keep top_k.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)  # (n,)
        order = np.argsort(-scores)[:top_k]
        return [(candidates[i], float(scores[i])) for i in order]

    def rerank_ids(self, query: str, id_text_pairs: list[tuple[str, str]], top_k: int = 5) -> list[tuple[str, float]]:
        """Same as rerank but returns the caller's ids alongside scores."""
        if not id_text_pairs:
            return []
        pairs = [[query, t] for _, t in id_text_pairs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(-scores)[:top_k]
        return [(id_text_pairs[i][0], float(scores[i])) for i in order]
