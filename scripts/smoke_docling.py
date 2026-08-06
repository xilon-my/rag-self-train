"""D1 parse smoke: run Docling on 3 representative PDFs and report block stats."""
import json
import sys
import time

sys.path.insert(0, "src")
from ingest.docling_parse import parse_pdf  # noqa: E402
from ingest.chunk import chunk_document  # noqa: E402

PDFS = sys.argv[1:]  # 3 pdf paths


def main():
    t0 = time.time()
    for p in PDFS:
        t1 = time.time()
        res = parse_pdf(p, "/tmp/docling_images")
        ntext = sum(1 for b in res["blocks"] if b["type"] == "text")
        ntable = sum(1 for b in res["blocks"] if b["type"] == "table")
        nimg = sum(1 for b in res["blocks"] if b["type"] == "image")
        chunks = chunk_document(res["pdf"], res["blocks"])
        print(f"[{p.split('/')[-1][:40]}] {time.time()-t1:.0f}s | "
              f"text={ntext} table={ntable} image={nimg} images_saved={len(res['images'])} | chunks={len(chunks)} | md_chars={len(res['md'])}")
        # sample: first table markdown + first image path
        for b in res["blocks"]:
            if b["type"] == "table":
                print("  TABLE sample:", b["text"][:120].replace("\n", " "))
                break
        for b in res["blocks"]:
            if b["type"] == "image":
                print("  IMAGE path:", b["img_path"])
                break
        # save a small sample of chunks for inspection
        with open(f"/tmp/docling_smoke_{res['pdf'].split('/')[-1][:20]}.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks[:5]], f, ensure_ascii=False, indent=1)
    print(f"TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
