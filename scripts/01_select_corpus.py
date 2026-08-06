"""Select a stratified subset (~40) of the FinRAGBench-V Chinese financial PDFs
and write corpus/MANIFEST.csv. Categories:
  company  — annual/company reports (tables-heavy: financial statements)
  research — industry/securities research (charts-heavy)
  policy   — monetary-policy / macro / statistical bulletins (text-heavy)
  news     — daily market news / commentary

Selection is deterministic (fixed seed). Prefers higher page counts within each
category so every doc is substantial. Source URL + license recorded per row.
"""
from __future__ import annotations

import csv
import os
import random
import re
import sys

import fitz  # pymupdf

CORPUS_DIR = sys.argv[1] if len(sys.argv) > 1 else "corpus/pdfs"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else "corpus/MANIFEST.csv"
TARGET = 40
SEED = 42
DATASET = "zhaosuifeng/FinRAGBench-V"
LICENSE = "apache-2.0"

CATEGORY_RULES = [
    ("company", r"(年度报告|年报|__\d{6}__)"),
    ("research", r"(行业|研究|洞察|策略|专题|展望|发展报告|分析)"),
    ("policy", r"(货币政策|统计公报|宏观观察|季报|汇率|经济观察|区域金融运行)"),
    ("news", r"(东方财富|中国金融信息网|晚报|日报|A股|黄金|股市|解禁|分红|发行)"),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, name):
            return cat
    return "research"


def main():
    pdfs = sorted(
        os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")
    )
    print(f"scanning {len(pdfs)} pdfs for page counts ...")
    rows = []
    for p in pdfs:
        try:
            with fitz.open(p) as doc:
                pages = doc.page_count
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        rows.append({
            "id": f"doc_{len(rows):04d}",
            "filename": os.path.basename(p),
            "path": p,
            "pages": pages,
            "size_mb": round(os.path.getsize(p) / 1e6, 2),
            "category": categorize(os.path.basename(p)),
        })

    # stratified selection: prefer bigger docs within each category
    rng = random.Random(SEED)
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    print("category counts:", {k: len(v) for k, v in by_cat.items()})

    # target split: company 10, research 12, policy 8, news 10
    targets = {"company": 10, "research": 12, "policy": 8, "news": 10}
    selected = []
    for cat, n in targets.items():
        pool = sorted(by_cat.get(cat, []), key=lambda r: (-r["pages"], r["filename"]))
        if len(pool) < n:
            pool = by_cat.get(cat, [])
        chosen = pool[:n]
        selected.extend(chosen)
        print(f"  {cat}: picked {len(chosen)}/{len(pool)}")

    # fill shortfall from the biggest remaining docs of any category
    if len(selected) < TARGET:
        chosen_ids = {r["id"] for r in selected}
        rest = sorted((r for r in rows if r["id"] not in chosen_ids), key=lambda r: -r["pages"])
        for r in rest[: TARGET - len(selected)]:
            selected.append(r)

    selected.sort(key=lambda r: r["id"])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "filename", "pages", "size_mb", "category", "source", "license"])
        for r in selected:
            w.writerow([r["id"], r["filename"], r["pages"], r["size_mb"], r["category"],
                        f"hf://datasets/{DATASET}/pdfs_for_QA", LICENSE])

    total_pages = sum(r["pages"] for r in selected)
    print(f"\nselected {len(selected)} pdfs, {total_pages} pages total, "
          f"{sum(r['size_mb'] for r in selected):.1f} MB")
    print("category mix:", {c: sum(1 for r in selected if r['category'] == c) for c in by_cat})
    print(f"MANIFEST -> {MANIFEST}")


if __name__ == "__main__":
    main()
