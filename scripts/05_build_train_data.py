"""Build (query, positive page, hard-negative pages) triples for training.

Positives: benchmark qrels (query -> gold page). Hard negatives: BM25 top pages
that are NOT the gold page (semantically close, same doc or same topic). This
gives the reranker meaningful negatives and the bi-encoder hard in-batch signal.

Output: data/train_triples.jsonl  {query, pos, neg: [..], pos_id, neg_ids}
"""
import json
import sys

sys.path.insert(0, "src")
from retrieve.bm25 import BM25Index  # noqa: E402

DATA = "data"


def main():
    qrel = {}
    for line in open(f"{DATA}/qrels_ch.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            qrel.setdefault(p[0], []).append(p[1])

    pages = []
    for l in open(f"{DATA}/corpus_pages.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["text"]:
            pages.append((r["corpus_id"], r["text"]))
    ids = [p[0] for p in pages]
    texts = [p[1] for p in pages]
    bm25 = BM25Index(ids, texts, tokenizer="char_bigram")
    id2text = {i: t for i, t in zip(ids, texts)}
    print(f"corpus {len(pages)} pages, building BM25")

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
        # hard negatives: BM25 top-10 excluding gold and same-doc
        gold_doc = pos.rsplit("_", 1)[0]
        cands = [c for c, _ in bm25.search(q["query"], top_k=30)
                 if c != pos and c.rsplit("_", 1)[0] != gold_doc]
        negs = cands[:3]
        if not negs:
            continue
        out.append({
            "query": q["query"],
            "pos": id2text[pos],
            "pos_id": pos,
            "neg": [id2text[n] for n in negs],
            "neg_ids": negs,
        })
        used += 1
        if used % 100 == 0:
            print(f"  {used}", flush=True)

    with open(f"{DATA}/train_triples.jsonl", "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"train triples: {len(out)}")


if __name__ == "__main__":
    main()
