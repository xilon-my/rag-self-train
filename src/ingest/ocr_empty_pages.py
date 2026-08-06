"""OCR the image-only corpus pages (no extractable text) with RapidOCR.

For each page whose pymupdf text is empty: render the page to an image and OCR
it, then attach the OCR text. Scanned magazines get full text; chart pages get
axis labels/titles/legend — enough to make the page text-retrievable.

Multiprocessing over workers (CPU). Writes an updated corpus_pages.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import fitz
from rapidocr_onnxruntime import RapidOCR

PDFS_DIR = sys.argv[1] if len(sys.argv) > 1 else "corpus/pdfs_selected_for_queries"
CORPUS = sys.argv[2] if len(sys.argv) > 2 else "data/corpus_pages.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/corpus_pages_ocr.jsonl"
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
MAX_PIXELS = 2200  # cap render size for OCR speed


def ocr_one(args):
    pdf_path, page_no, corpus_id = args
    try:
        with fitz.open(pdf_path) as d:
            page = d[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            # cap to MAX_PIXELS on the long side
            if max(pix.width, pix.height) > MAX_PIXELS:
                scale = MAX_PIXELS / max(pix.width, pix.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5 * scale, 1.5 * scale))
            img = pix.tobytes("png")
        result, _ = RapidOCR()(img)
        text = "\n".join(line[1] for line in result) if result else ""
        return corpus_id, text
    except Exception as e:
        return corpus_id, f"[OCR_ERROR: {e}]"


def main():
    # load corpus, find docs+pages needing OCR
    doc2pdf = {os.path.splitext(f)[0]: os.path.join(PDFS_DIR, f)
               for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")}
    tasks = []
    keep = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["text"]:
                pdf = doc2pdf.get(r["doc"])
                if pdf:
                    tasks.append((pdf, r["page"], r["corpus_id"]))
            keep.append(r)
    print(f"pages to OCR: {len(tasks)}", flush=True)

    ocr_results = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (cid, text) in enumerate(ex.map(partial(ocr_one), tasks)):
            ocr_results[cid] = text
            if (i + 1) % 500 == 0:
                print(f"  ocr'd {i+1}/{len(tasks)}", flush=True)

    # merge
    n_filled = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for r in keep:
            if not r["text"]:
                ocr = ocr_results.get(r["corpus_id"], "")
                if ocr:
                    r["text"] = ocr
                    n_filled += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"filled {n_filled} pages with OCR text -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
