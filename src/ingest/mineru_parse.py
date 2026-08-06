"""MinerU (magic-pdf) wrapper: parse one PDF into structured blocks and write a
normalized JSONL with the content_list blocks (text / table-markdown / image path).

Run via the magic-pdf CLI (most stable API surface), then read back
content_list.json. Kept deliberately thin — the CLI does the heavy lifting.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def parse_pdf(pdf_path: str, out_dir: str, method: str = "auto", timeout: int = 1800) -> dict:
    """Parse one PDF with magic-pdf CLI. Returns the parsed record:
    {pdf, blocks: [{type, text|img_path, page}], images: [paths]}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["magic-pdf", "-p", pdf_path, "-o", str(out_dir), "-m", method]
    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)

    # CLI writes out_dir/<docname>/content_list.json
    docname = Path(pdf_path).stem
    result_dir = out_dir / docname
    content_file = result_dir / "content_list.json"
    if not content_file.exists():
        # fallback: scan for any content_list.json under out_dir
        cands = list(out_dir.rglob("content_list.json"))
        if not cands:
            raise FileNotFoundError(f"content_list.json not found for {pdf_path} under {out_dir}")
        content_file = cands[0]
        result_dir = content_file.parent

    with open(content_file, encoding="utf-8") as f:
        blocks = json.load(f)

    images = []
    for b in blocks:
        img = b.get("img_path")
        if img and Path(str(result_dir / img)).exists():
            images.append(str(result_dir / img))

    return {"pdf": pdf_path, "blocks": blocks, "images": images, "result_dir": str(result_dir)}
