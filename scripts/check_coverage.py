"""Validate that the benchmark's Chinese queries map onto the PDFs we hold.

query-id format is '{docname}_{page}.png-{n}'. We compare {docname} against the
PDF filenames (stripped of '.pdf'). Reports how many queries we can use with a
given set of PDFs.
"""
import json
import os
import sys

queries_path = sys.argv[1]
pdfs_dir = sys.argv[2]

with open(queries_path, encoding="utf-8") as f:
    queries = json.load(f)

pdf_names = {os.path.splitext(f)[0] for f in os.listdir(pdfs_dir) if f.endswith(".pdf")}


def doc_of(qid: str) -> str:
    # strip trailing '.png-N' (possibly multiple .png markers)
    return qid.split("_")[0]  # heuristic; refined below


def doc_of_v2(qid: str) -> str:
    # qid ends with '_<page>.png-<n>' OR '_<page>.png'; remove that suffix
    import re
    m = re.search(r"_(?:\d+)\.png(?:-\d+)?$", qid)
    if m:
        return qid[: m.start()]
    return qid


# try both heuristics, count overlap with our pdfs
for fn_name, fn in [("doc_of", doc_of), ("doc_of_v2", doc_of_v2)]:
    docs = set()
    matched = 0
    examples = []
    for q in queries:
        d = fn(q["query-id"])
        docs.add(d)
        if d in pdf_names:
            matched += 1
            if len(examples) < 3:
                examples.append((d, q["query"][:40]))
    print(f"[{fn}] unique docs={len(docs)} | matched queries={matched}/{len(queries)}")
    for d, q in examples:
        print(f"    {d!r} <- {q}...")
    print(f"    sample doc ids:", list(docs)[:5])
