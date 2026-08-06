"""D3 baseline: run off-the-shelf retrieval rows on the 99 golden queries.

Rows:
  1. BM25 only (char-bigram)
  2. BM25 + dense (frozen bge-base-zh-v1.5) + RRF
  3. Row 2 + rerank top-50->top-5 with frozen bge-reranker-base

Metrics (page-level, single-gold per query): Recall@10 (== Hit@10), nDCG@10, MRR@10.
Plain numbers — no CI/Wilcoxon machinery (the honest, lean eval the user chose).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from retrieve.bm25 import BM25Index  # noqa: E402
from retrieve.dense import DenseIndex  # noqa: E402
from retrieve.fusion import rrf_fuse  # noqa: E402
from retrieve.query_encoder import QueryEncoder  # noqa: E402
from retrieve.rerank import CrossEncoderReranker  # noqa: E402

DATA = "data"


def load_golden():
    rows = [json.loads(l) for l in open(f"{DATA}/golden.jsonl", encoding="utf-8")]
    # qrels: query-id -> [corpus_id]
    qrel = {}
    for line in open(f"{DATA}/qrels_ch.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            qrel.setdefault(p[0], []).append(p[1])
    return rows, qrel


def load_corpus():
    pages = []
    for l in open(f"{DATA}/corpus_pages.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["text"]:
            pages.append((r["corpus_id"], r["text"]))
    return pages


def ndcg10(rank):
    # single relevant doc at 1-based rank
    if rank is None:
        return 0.0
    if rank == 1:
        return 1.0
    return 1.0 / np.log2(rank + 1)  # nDCG = DCG/IDCG = (1/log2(rank+1))/1


def eval_row(order, qrel, qid):
    """order: ranked list of corpus_ids. Returns (r10, ndcg10, mrr)."""
    golds = set(qrel.get(qid, []))
    if not golds:
        return None
    top10 = order[:10]
    r10 = 1.0 if any(g in top10 for g in golds) else 0.0
    rank = None
    for i, cid in enumerate(order):
        if cid in golds:
            rank = i + 1
            break
    return r10, ndcg10(rank), (1.0 / rank if rank else 0.0)


def main():
    t0 = time.time()
    pages = load_corpus()
    ids = [p[0] for p in pages]
    texts = [p[1] for p in pages]
    print(f"corpus: {len(pages)} pages ({time.time()-t0:.0f}s)")

    print("building BM25 (char-bigram) ...")
    bm25 = BM25Index(ids, texts, tokenizer="char_bigram")

    print("encoding pages with frozen bge-base-zh-v1.5 (GPU) ...")
    enc = QueryEncoder("BAAI/bge-base-zh-v1.5", device="cuda")
    vecs = enc.encode(texts, is_query=False)
    dense = DenseIndex(ids, vecs)
    np.save(f"{DATA}/dense_offshell.npy", vecs)

    print("loading reranker (frozen bge-reranker-base) ...")
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base", device="cuda")

    rows, qrel = load_golden()
    print(f"golden queries: {len(rows)}")

    agg = {"r10": [], "ndcg": [], "mrr": []}
    for name, topk in [("1_bm25", 50), ("2_dense_rrf", 50)]:
        res = {"r10": [], "ndcg": [], "mrr": []}
        for q in rows:
            qid = q["query-id"]
            if qid not in qrel:
                continue
            if name == "1_bm25":
                order = [c for c, _ in bm25.search(q["query"], top_k=topk)]
            else:
                bm25_rank = bm25.search(q["query"], top_k=topk)
                d_rank = dense.search(enc.encode([q["query"]], is_query=True)[0], top_k=topk)
                order = [c for c, _ in rrf_fuse([[c for c, _ in bm25_rank], [c for c, _ in d_rank]], k=60)]
            m = eval_row(order, qrel, qid)
            if m:
                res["r10"].append(m[0]); res["ndcg"].append(m[1]); res["mrr"].append(m[2])
        for k in res:
            agg[k] = res[k]
        print(f"  row {name}: R@10={np.mean(res['r10']):.3f} nDCG@10={np.mean(res['ndcg']):.3f} MRR={np.mean(res['mrr']):.3f} (n={len(res['r10'])})")

    # row 3: RRF top-50 -> rerank top-10
    res3 = {"r10": [], "ndcg": [], "mrr": []}
    for q in rows:
        qid = q["query-id"]
        if qid not in qrel:
            continue
        bm25_rank = bm25.search(q["query"], top_k=50)
        d_rank = dense.search(enc.encode([q["query"]], is_query=True)[0], top_k=50)
        fused = [c for c, _ in rrf_fuse([[c for c, _ in bm25_rank], [c for c, _ in d_rank]], k=60)]
        id_text_pairs = [(c, texts[ids.index(c)]) for c in fused[:50]]
        order_ids = [cid for cid, _ in reranker.rerank_ids(q["query"], id_text_pairs, top_k=10)]
        m = eval_row(order_ids, qrel, qid)
        if m:
            res3["r10"].append(m[0]); res3["ndcg"].append(m[1]); res3["mrr"].append(m[2])
    print(f"  row 3_full_rerank: R@10={np.mean(res3['r10']):.3f} nDCG@10={np.mean(res3['ndcg']):.3f} MRR={np.mean(res3['mrr']):.3f} (n={len(res3['r10'])})")

    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
