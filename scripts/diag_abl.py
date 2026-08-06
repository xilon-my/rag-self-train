import json
import sys

import numpy as np

sys.path.insert(0, "src")
from retrieve.bm25 import BM25Index
from retrieve.dense import DenseIndex
from retrieve.fusion import rrf_fuse
from retrieve.query_encoder import QueryEncoder
from retrieve.rerank import CrossEncoderReranker

DATA = "data"
rows = [json.loads(l) for l in open(f"{DATA}/golden.jsonl", encoding="utf-8")][:3]
pages = [(r["corpus_id"], r["text"]) for r in (json.loads(l) for l in open(f"{DATA}/corpus_pages.jsonl", encoding="utf-8")) if r["text"]]
ids = [p[0] for p in pages]; texts = [p[1] for p in pages]
bm25 = BM25Index(ids, texts, tokenizer="char_bigram")
off_enc = QueryEncoder("BAAI/bge-base-zh-v1.5", device="cuda")
dense_off = DenseIndex(ids, off_enc.encode(texts, is_query=False))
tr_enc = QueryEncoder("checkpoints/biencoder", device="cuda")
dense_tr = DenseIndex(ids, tr_enc.encode(texts, is_query=False))
rrk_off = CrossEncoderReranker("BAAI/bge-reranker-base", device="cuda")

for q in rows:
    qtext = q["query"]
    b_ids = [c for c, _ in bm25.search(qtext, top_k=50)]
    d_off = dense_off.search(off_enc.encode([qtext], is_query=True)[0], top_k=50)
    d_tr = dense_tr.search(tr_enc.encode([qtext], is_query=True)[0], top_k=50)
    r2 = [c for c, _ in rrf_fuse([b_ids, [c for c, _ in d_off]], k=60)][:10]
    r4 = [c for c, _ in rrf_fuse([b_ids, [c for c, _ in d_tr]], k=60)][:10]
    pairs = [(c, texts[ids.index(c)]) for c in r4[:50]]
    r3 = [cid for cid, _ in rrk_off.rerank_ids(qtext, pairs[:50], top_k=10)]
    print(f"\nQ: {q['query'][:40]}")
    print("  R2(off_rrf)   :", r2[:5])
    print("  R3(off_rerank):", r3[:5])
    print("  R4(tr_rrf)    :", r4[:5])
    print("  r2==r3?", r2 == r3, "| r2==r4?", r2 == r4, "| r3==r4?", r3 == r4)
