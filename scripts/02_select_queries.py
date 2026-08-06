"""Select golden (eval) + training queries from FinRAGBench-V, doc-level split.

Uses qrels (query-id -> corpus-id) to derive the source document: corpus-id is
'{doc}_{page}.png', so stripping the trailing '_{page}.png' gives the doc name.
Queries whose doc is not among our PDFs are dropped. Docs are split train/eval
at doc level (no doc appears in both). Outputs:
  data/golden.jsonl      eval queries (target ~120, category-diverse)
  data/train_pairs.jsonl training (query, doc, relevant_page) tuples
  data/needed_docs.txt   unique docs needed for the corpus builder
"""
import json
import os
import random
import re
import sys

QUERIES = sys.argv[1] if len(sys.argv) > 1 else "data/queries_ch.json"
QRELS = sys.argv[2] if len(sys.argv) > 2 else "data/qrels_ch.tsv"
PDFS_DIR = sys.argv[3] if len(sys.argv) > 3 else "corpus/pdfs_selected_for_queries"
N_GOLDEN = int(sys.argv[4]) if len(sys.argv) > 4 else 120
SEED = 42

_PAGE_RE = re.compile(r"_\d+(?:-\d+)?\.png$")


def doc_from_corpus_id(cid: str) -> str:
    return _PAGE_RE.sub("", cid)


def load_qrels(path):
    qrel = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            qrel.setdefault(parts[0], []).append(parts[1])
    return qrel


def main():
    with open(QUERIES, encoding="utf-8") as f:
        queries = json.load(f)
    qrel = load_qrels(QRELS)

    pdf_names = {os.path.splitext(f)[0] for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")}

    # keep queries with a known corpus-id
    rows = []
    for q in queries:
        qid = q["query-id"]
        if qid not in qrel:
            continue
        cid = qrel[qid][0]
        doc = doc_from_corpus_id(cid)
        # match doc against our PDFs (also try stripping '_multipage')
        if doc in pdf_names:
            key = doc
        else:
            alt = doc.replace("_multipage", "")
            if alt in pdf_names:
                key = alt
            else:
                continue
        rows.append({"query-id": qid, "query": q["query"], "answer": q.get("answer", ""),
                     "category": q.get("category", ""), "corpus-id": cid, "doc": key})

    print(f"usable queries: {len(rows)}/{len(queries)} across {len({r['doc'] for r in rows})} docs")

    # doc-level split
    rng = random.Random(SEED)
    docs = sorted({r["doc"] for r in rows})
    rng.shuffle(docs)
    n_eval_docs = max(1, int(len(docs) * 0.25))
    eval_docs = set(docs[:n_eval_docs])
    train_docs = set(docs[n_eval_docs:])

    eval_rows = [r for r in rows if r["doc"] in eval_docs]
    train_rows = [r for r in rows if r["doc"] in train_docs]

    # category-diverse golden selection
    by_cat: dict[str, list] = {}
    for r in eval_rows:
        by_cat.setdefault(r["category"], []).append(r)
    golden = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda r: r["query-id"])
        rng.shuffle(pool)
        golden.extend(pool[: max(1, N_GOLDEN // len(by_cat))])
    golden = golden[:N_GOLDEN]

    with open("data/golden.jsonl", "w", encoding="utf-8") as f:
        for r in golden:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/train_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/needed_docs.txt", "w", encoding="utf-8") as f:
        for d in sorted(eval_docs | train_docs):
            f.write(d + "\n")

    print(f"eval docs: {len(eval_docs)} | train docs: {len(train_docs)}")
    print(f"golden queries: {len(golden)} | training queries: {len(train_rows)}")
    print(f"needed docs (total): {len(eval_docs | train_docs)}")


if __name__ == "__main__":
    main()
