"""Build a page-level retrieval corpus from our PDFs using pymupdf.

Each record: {corpus_id: '{doc}_{page}.png', doc, page, text} — the corpus_id
format matches FinRAGBench-V's qrels so benchmark queries map cleanly.

We store TEXT only (pymupdf page.get_text). Page images are rendered on demand
in the demo (pymupdf is fast) so we don't burn tens of GB on rendered pages.
"""
from __future__ import annotations

import json
import os
import sys

import fitz  # pymupdf

PDFS_DIR = sys.argv[1] if len(sys.argv) > 1 else "corpus/pdfs_selected_for_queries"
NEEDED = sys.argv[2] if len(sys.argv) > 2 else "data/needed_docs.txt"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/corpus_pages.jsonl"


def main():
    with open(NEEDED, encoding="utf-8") as f:
        needed = {line.strip() for line in f if line.strip()}

    # doc name -> pdf path
    doc2pdf = {}
    for fn in os.listdir(PDFS_DIR):
        if fn.endswith(".pdf"):
            doc2pdf[os.path.splitext(fn)[0]] = os.path.join(PDFS_DIR, fn)

    missing = [d for d in needed if d not in doc2pdf]
    if missing:
        print(f"WARN missing {len(missing)} pdfs, e.g. {missing[:5]}")

    n_docs = n_pages = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for doc in sorted(needed):
            pdf = doc2pdf.get(doc)
            if pdf is None:
                continue
            try:
                with fitz.open(pdf) as d:
                    n_pages_in_doc = d.page_count
                    for i, page in enumerate(d, start=1):
                        text = page.get_text("text").strip()
                        rec = {
                            "corpus_id": f"{doc}_{i}.png",
                            "doc": doc,
                            "page": i,
                            "text": text,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_docs += 1
                n_pages += n_pages_in_doc
            except Exception as e:
                print(f"  fail {doc}: {e}")
    print(f"built corpus: {n_docs} docs, {n_pages} pages -> {OUT}")


if __name__ == "__main__":
    main()
