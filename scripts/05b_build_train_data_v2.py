"""Improved training triples: same-doc + cross-doc + self-mined hard negatives.

The v1 negatives excluded same-doc pages, but at inference the fused top-50 is
full of the gold doc's other pages — the reranker never learned to reject them.
v2 adds (per query, up to 8 negatives):
  1. same-doc pages (hardest: topically closest)
  2. cross-doc BM25 near-misses
  3. self-mined: trained bi-encoder's top non-gold pages (the model's own confusions)

Output: data/train_triples_v2.jsonl
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from retrieve.bm25 import BM25Index  # noqa: E402
from retrieve.dense import DenseIndex  # noqa: E402
from retrieve.query_encoder import QueryEncoder  # noqa: E402

DATA = "data"
N_SAME = 3
N_CROSS = 3
N_SELF = 2


def main():
    qrel = {}
    for line in open(f"{DATA}/qrels_ch.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            qrel.setdefault(p[0], []).append(p[1])

    pages = [(r["corpus_id"], r["text"]) for r in
             (json.loads(l) for l in open(f"{DATA}/corpus_pages.jsonl", encoding="utf-8"))
             if r["text"]]
    ids = [p[0] for p in pages]
    texts = [p[1] for p in pages]
    id2text = dict(zip(ids, texts))
    # doc -> page ids (for same-doc negatives)
    doc_pages: dict[str, list[str]] = {}
    for cid in ids:
        doc_pages.setdefault(cid.rsplit("_", 1)[0], []).append(cid)

    print("building BM25 ...")
    bm25 = BM25Index(ids, texts, tokenizer="char_bigram")
    print("encoding with trained bi-encoder for self-mining ...")
    enc = QueryEncoder("checkpoints/biencoder", device="cuda")
    vecs = enc.encode(texts, is_query=False)
    dense = DenseIndex(ids, vecs)

    train = [json.loads(l) for l in open(f"{DATA}/train_pairs.jsonl", encoding="utf-8")]
    out = []
    used = 0
    for q in train:
        qid = q["query-id"]
        golds = qrel.get(qid, [])
        if not golds:
            continue
        pos = golds[0]
        if pos not in id2text:
            continue
        gold_doc = pos.rsplit("_", 1)[0]
        qvec = enc.encode([q["query"]], is_query=True)[0]

        # 1) same-doc negatives: pages of the gold doc except the gold page
        same = [c for c in doc_pages.get(gold_doc, []) if c != pos][: N_SAME]
        # 2) cross-doc BM25 near-misses
        cross = [c for c, _ in bm25.search(q["query"], top_k=40)
                 if c != pos and c.rsplit("_", 1)[0] != gold_doc][: N_CROSS]
        # 3) self-mined: trained bi-encoder top non-gold (skip same-doc to diversify)
        selfm = [c for c, _ in dense.search(qvec, top_k=20)
                 if c != pos and c.rsplit("_", 1)[0] != gold_doc][: N_SELF]

        negs = list(dict.fromkeys(same + cross + selfm))
        if not negs:
            continue
        out.append({
            "query": q["query"], "pos": id2text[pos], "pos_id": pos,
            "neg": [id2text[n] for n in negs], "neg_ids": negs,
        })
        used += 1
        if used % 100 == 0:
            print(f"  {used}", flush=True)

    with open(f"{DATA}/train_triples_v2.jsonl", "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n_neg = sum(len(t["neg"]) for t in out)
    print(f"triples: {len(out)}, avg neg/query: {n_neg/len(out):.1f}")


if __name__ == "__main__":
    main()
