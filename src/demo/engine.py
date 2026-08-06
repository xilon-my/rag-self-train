"""Minimal demo engine: load corpus + trained models, retrieve top pages for a
query, render the page image, optionally generate an answer.

Runs on CPU if no GPU. Expects data/ (corpus_pages.jsonl) and checkpoints/
(biencoder, reranker_v2) — restore from the GitHub Release tarball if missing.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from retrieve.bm25 import BM25Index  # noqa: E402
from retrieve.dense import DenseIndex  # noqa: E402
from retrieve.fusion import rrf_fuse  # noqa: E402
from retrieve.query_encoder import QueryEncoder  # noqa: E402
from retrieve.rerank import CrossEncoderReranker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DemoEngine:
    def __init__(self, device="auto", top_k_retrieve=50, top_k_final=5):
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

        # corpus
        pages = []
        for l in open(os.path.join(ROOT, "data", "corpus_pages.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            if r["text"]:
                pages.append((r["corpus_id"], r["text"]))
        self.ids = [p[0] for p in pages]
        self.texts = [p[1] for p in pages]
        self.bm25 = BM25Index(self.ids, self.texts, tokenizer="char_bigram")

        # trained models
        self.enc = QueryEncoder(os.path.join(ROOT, "checkpoints", "biencoder"), device=device)
        vecs = self.enc.encode(self.texts, is_query=False)
        self.dense = DenseIndex(self.ids, vecs)
        self.reranker = CrossEncoderReranker(os.path.join(ROOT, "checkpoints", "reranker_v2"), device=device)

    def retrieve(self, query: str):
        b_ids = [c for c, _ in self.bm25.search(query, top_k=self.top_k_retrieve)]
        d_rank = self.dense.search(self.enc.encode([query], is_query=True)[0], top_k=self.top_k_retrieve)
        fused = [c for c, _ in rrf_fuse([b_ids, [c for c, _ in d_rank]], k=60)]
        pairs = [(c, self.texts[self.ids.index(c)]) for c in fused[: self.top_k_retrieve]]
        reranked = self.reranker.rerank_ids(query, pairs, top_k=self.top_k_final)
        out = []
        for cid, score in reranked:
            i = self.ids.index(cid)
            out.append({"corpus_id": cid, "score": score, "text": self.texts[i],
                        "doc": cid.rsplit("_", 1)[0], "page": int(cid.rsplit("_", 1)[1].rstrip(".png"))})
        return out

    def render_page(self, doc: str, page: int, out_path: str) -> str | None:
        import fitz
        pdf = os.path.join(ROOT, "corpus", "pdfs_selected_for_queries", f"{doc}.pdf")
        if not os.path.exists(pdf):
            return None
        with fitz.open(pdf) as d:
            pix = d[page - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            pix.save(out_path)
        return out_path
