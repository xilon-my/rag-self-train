"""Targeted MinerU OCR for query-referenced image-only pages.

Only the ~133 answer pages (golden + training queries' gold pages) that have no
extractable text are OCR'd. For each doc: extract just those pages into a temp
PDF (pymupdf), run magic-pdf on it, read the OCR text per page, and write an
updated corpus_pages.jsonl where those pages now carry text.

Output: data/corpus_pages.jsonl (in place, text filled in).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import fitz

PDFS_DIR = sys.argv[1] if len(sys.argv) > 1 else "corpus/pdfs_selected_for_queries"
QUERY_FILES = sys.argv[2:] if len(sys.argv) > 2 else ["data/golden.jsonl", "data/train_pairs.jsonl"]
CORPUS = "data/corpus_pages.jsonl"
TMPDIR = "/tmp/mp_ocr"


def load_qrels():
    qrel = {}
    for line in open("data/qrels_ch.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            qrel.setdefault(p[0], []).append(p[1])
    return qrel


def main():
    qrel = load_qrels()
    # answer pages that need OCR = queries' gold page AND empty in current corpus
    text_ids = set()
    corpus = []
    for l in open(CORPUS, encoding="utf-8"):
        r = json.loads(l)
        corpus.append(r)
        if r["text"]:
            text_ids.add(r["corpus_id"])

    need = {}  # doc -> set of page numbers (1-based)
    for qf in QUERY_FILES:
        for l in open(qf, encoding="utf-8"):
            q = json.loads(l)
            g = qrel.get(q["query-id"], [])
            if g and g[0] not in text_ids:
                cid = g[0]
                # corpus_id = '{doc}_{page}.png'
                doc, page = cid.rsplit("_", 1)
                page = int(page[:-4])  # strip '.png'
                need.setdefault(doc, set()).add(page)

    print(f"docs with image-only answer pages: {len(need)}, pages: {sum(len(v) for v in need.values())}")
    os.makedirs(TMPDIR, exist_ok=True)

    doc2pdf = {os.path.splitext(f)[0]: os.path.join(PDFS_DIR, f)
               for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")}

    filled = {}
    for doc, pages in sorted(need.items()):
        pdf = doc2pdf.get(doc)
        if pdf is None:
            print(f"  skip missing {doc}")
            continue
        pg = sorted(pages)
        tmp = os.path.join(TMPDIR, f"{doc[:40]}.pdf")
        with fitz.open(pdf) as d:
            out = fitz.open()
            for p in pg:
                out.insert_pdf(d, from_page=p - 1, to_page=p - 1)
            out.save(tmp)
        # map temp page index -> original page
        order = pg
        base = os.path.splitext(os.path.basename(tmp))[0]
        # resumable: reuse content_list if it already exists
        cands = glob.glob(os.path.join(TMPDIR, "**", "*_content_list.json"), recursive=True)
        cands = [c for c in cands if os.path.basename(c).startswith(base)]
        if not cands:
            magic_pdf_bin = os.path.join(os.path.dirname(sys.executable), "magic-pdf")
            res = subprocess.run(
                [magic_pdf_bin, "-p", tmp, "-o", TMPDIR, "-m", "auto"],
                capture_output=True, text=True, timeout=1800,
            )
            if res.returncode != 0:
                print(f"  magic-pdf fail {doc}: {res.stderr[-200:]}")
                continue
            cands = glob.glob(os.path.join(TMPDIR, "**", "*_content_list.json"), recursive=True)
            cands = [c for c in cands if os.path.basename(c).startswith(base)]
        if not cands:
            print(f"  no content_list for {doc}")
            continue
        with open(cands[0], encoding="utf-8") as f:
            blocks = json.load(f)
        page_texts = {}
        for b in blocks:
            pi = b.get("page_idx")
            if pi is not None and pi < len(order):
                t = (b.get("text") or "").strip()
                page_texts.setdefault(pi, []).append(t)
        for i, orig in enumerate(order):
            text = "\n".join(page_texts.get(i, []))
            if text:
                filled[f"{doc}_{orig}.png"] = text
        print(f"  {doc}: {len(pg)} pages, {sum(1 for i in order if f'{doc}_{i}.png' in filled)} filled")

    # merge into corpus
    n = 0
    with open(CORPUS, "w", encoding="utf-8") as f:
        for r in corpus:
            if not r["text"] and r["corpus_id"] in filled:
                r["text"] = filled[r["corpus_id"]]
                n += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OCR'd {n} pages into corpus")


if __name__ == "__main__":
    main()
