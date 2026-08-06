"""BM25 from first principles (rank_bm25 backend), jieba + char-bigram tokenizers.

char-bigram often beats jieba on Chinese legal/financial phrasing because jieba
segments jargon (处...罚款, 招股说明书) poorly; both are implemented and the
winner is measured in the ablation.
"""
from __future__ import annotations

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, doc_ids: list[str], doc_texts: list[str],
                 tokenizer: str = "char_bigram", k1: float = 1.5, b: float = 0.75):
        assert tokenizer in {"jieba", "char_bigram"}, tokenizer
        self.tokenizer = tokenizer
        self.doc_ids = doc_ids
        self.tokenized_docs = [self.tokenize(t) for t in doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)

    def tokenize(self, text: str) -> list[str]:
        if self.tokenizer == "jieba":
            return [w for w in jieba.cut(text) if w.strip()]
        # char_bigram
        s = text.replace(" ", "").replace("\n", "")
        grams = [s[i:i + 2] for i in range(len(s) - 1)]
        return grams or list(s)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(self.tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]
