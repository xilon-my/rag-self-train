"""Full 5-row ablation: off-the-shelf vs self-trained retrieval stack.

Rows (same 99 golden queries, same code path, only weights differ):
  1. BM25 only
  2. BM25 + dense (frozen bge-base-zh-v1.5) + RRF
  3. Row 2 + rerank top-50->top-10 with frozen bge-reranker-base
  4. BM25 + dense (TRAINED bi-encoder) + RRF
  5. Row 4 + rerank with TRAINED bge-reranker-base

Metrics: Recall@10, nDCG@10, MRR@10. Plain numbers.
"""
import json
import os
import sys

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
    qrel = {}
    for line in open(f"{DATA}/qrels_ch.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            qrel.setdefault(p[0], []).append(p[1])
    return rows, qrel


def load_corpus():
    pages = [(r["corpus_id"], r["text"]) for r in
             (json.loads(l) for l in open(f"{DATA}/corpus_pages.jsonl", encoding="utf-8"))
             if r["text"]]
    return pages


def ndcg10(rank):
    if rank is None:
        return 0.0
    return 1.0 if rank == 1 else 1.0 / np.log2(rank + 1)


def metrics(order, qrel, qid):
    golds = set(qrel.get(qid, []))
    if not golds:
        return None
    top10 = order[:10]
    r10 = 1.0 if any(g in top10 for g in golds) else 0.0
    rank = next((i + 1 for i, c in enumerate(order) if c in golds), None)
    return r10, ndcg10(rank), (1.0 / rank if rank else 0.0)


def main():
    pages = load_corpus()
    ids = [p[0] for p in pages]
    texts = [p[1] for p in pages]
    print(f"corpus: {len(pages)}")
    bm25 = BM25Index(ids, texts, tokenizer="char_bigram")

    # dense indexes
    off_enc = QueryEncoder("BAAI/bge-base-zh-v1.5", device="cuda")
    vecs_off = off_enc.encode(texts, is_query=False)
    dense_off = DenseIndex(ids, vecs_off)

    trained_enc = QueryEncoder("checkpoints/biencoder", device="cuda")
    vecs_tr = trained_enc.encode(texts, is_query=False)
    dense_tr = DenseIndex(ids, vecs_tr)

    # rerankers
    rrk_off = CrossEncoderReranker("BAAI/bge-reranker-base", device="cuda")
    rrk_tr = CrossEncoderReranker("checkpoints/reranker_v2", device="cuda")

    rows, qrel = load_golden()
    print(f"golden: {len(rows)}")

    results = {name: {"r10": [], "ndcg": [], "mrr": []} for name in ["1_bm25", "2_off_rrf", "3_off_rerank", "4_tr_rrf", "5_tr_rerank"]}
    for q in rows:
        qid = q["query-id"]
        if qid not in qrel:
            continue
        qtext = q["query"]
        b_rank = bm25.search(qtext, top_k=50)
        b_ids = [c for c, _ in b_rank]

        def fused(enc, dense_idx):
            d_rank = dense_idx.search(enc.encode([qtext], is_query=True)[0], top_k=50)
            return [c for c, _ in rrf_fuse([b_ids, [c for c, _ in d_rank]], k=60)]

        def reranked(enc, dense_idx, rrk):
            fused_ids = fused(enc, dense_idx)[:50]
            pairs = [(c, texts[ids.index(c)]) for c in fused_ids]
            return [cid for cid, _ in rrk.rerank_ids(qtext, pairs, top_k=10)]

        orders = {
            "1_bm25": b_ids[:10],
            "2_off_rrf": fused(off_enc, dense_off)[:10],
            "3_off_rerank": reranked(off_enc, dense_off, rrk_off),
            "4_tr_rrf": fused(trained_enc, dense_tr)[:10],
            "5_tr_rerank": reranked(trained_enc, dense_tr, rrk_tr),
        }
        for name, order in orders.items():
            m = metrics(order, qrel, qid)
            if m:
                results[name]["r10"].append(m[0])
                results[name]["ndcg"].append(m[1])
                results[name]["mrr"].append(m[2])

    print("\n=== ABLATION ===")
    print(f"{'row':<16}{'R@10':>8}{'nDCG@10':>9}{'MRR':>8}")
    for name, res in results.items():
        if res["r10"]:
            print(f"{name:<16}{np.mean(res['r10']):>8.3f}{np.mean(res['ndcg']):>9.3f}{np.mean(res['mrr']):>8.3f}")

    with open("results/ablation.csv", "w", encoding="utf-8") as f:
        f.write("row,recall@10,ndcg@10,mrr\n")
        for name, res in results.items():
            if res["r10"]:
                f.write(f"{name},{np.mean(res['r10']):.4f},{np.mean(res['ndcg']):.4f},{np.mean(res['mrr']):.4f}\n")

    # per-query results for paired bootstrap / Wilcoxon
    with open("results/per_query.json", "w", encoding="utf-8") as f:
        json.dump({name: res for name, res in results.items()}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
