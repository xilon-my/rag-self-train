"""Docling (IBM, MIT) PDF parser wrapper — replaces MinerU.

Converts one PDF to structured blocks: text / table-markdown / image. Every
block carries page provenance (needed for citation and page→chunk mapping).
Images are saved to disk so the multimodal pipeline can describe them with Kimi.

Docling's TableFormer is strong on financial tables (our corpus), and the
layout model + text flow extraction run on CPU. Digital-born PDFs need no OCR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from docling.document_converter import DocumentConverter

TEXT_LABELS = {"text", "title", "section_header", "paragraph", "caption"}
TABLE_LABELS = {"table"}
IMAGE_LABELS = {"picture", "figure"}


def parse_pdf(pdf_path: str, images_dir: str) -> dict:
    """Parse one PDF. Returns {pdf, blocks: [...], images: [...], md: str}.
    blocks: {type: text|table|image, text|markdown|img_path, pages: [int]}"""
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    blocks = []
    images = []
    img_idx = 0
    for item, _level in doc.iterate_items():
        label = str(item.label).lower()
        pages = sorted({p.page_no for p in item.prov}) if item.prov else [0]

        if label in TEXT_LABELS:
            text = (item.text or "").strip()
            if text:
                blocks.append({"type": "text", "text": text, "pages": pages})

        elif label in TABLE_LABELS:
            try:
                md = item.export_to_markdown()
            except Exception:
                md = str(item.text or "")
            if md.strip():
                blocks.append({"type": "table", "text": md, "pages": pages})

        elif label in IMAGE_LABELS:
            img = getattr(item, "image", None)
            if img is not None:
                img_idx += 1
                img_path = images_dir / f"{pdf_path.stem}_img{img_idx:03d}.png"
                img.save(img_path)
                images.append(str(img_path))
                blocks.append({
                    "type": "image",
                    "text": f"[图片 {img_idx}]",
                    "img_path": str(img_path),
                    "pages": pages,
                })

    md = doc.export_to_markdown()
    return {"pdf": str(pdf_path), "blocks": blocks, "images": images, "md": md}
