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

import torch  # noqa: E402

from docling.datamodel.settings import settings  # noqa: E402

# torch.compile's inductor worker crashes on this torch 2.7 nightly (sm_120); run eager.
settings.inference.compile_torch_models = False

import docling.models.inference_engines.object_detection.transformers_engine as _te  # noqa: E402
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# --- Workaround: RT-DETR (docling-layout-old) + transformers >= 4.57 padding bug ---
# Docling calls the HF image processor without `size`, so differently-sized page
# images can't form a batch tensor ("activate padding"). Pass a fixed longest-edge
# so all batch images resize to the same dims; post-processing still maps back via
# native `target_sizes`. Patched at import time, before any converter is built.
_orig_predict_batch = _te.TransformersObjectDetectionEngine.predict_batch


def _patched_predict_batch(self, input_batch):
    if not input_batch:
        return []
    images = [item.image.convert("RGB") for item in input_batch]
    inputs = self._processor(images=images, return_tensors="pt",
                             size={"height": 1280, "width": 1280}).to(self._device)
    target_sizes = torch.tensor([[img.height, img.width] for img in images], device=self._device)
    with torch.inference_mode():
        outputs = self._model(**inputs)
    results = self._processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=self.options.score_threshold
    )
    return results


_te.TransformersObjectDetectionEngine.predict_batch = _patched_predict_batch

TEXT_LABELS = {"text", "title", "section_header", "paragraph", "caption"}
TABLE_LABELS = {"table"}
IMAGE_LABELS = {"picture", "figure"}


def parse_pdf(pdf_path: str, images_dir: str) -> dict:
    """Parse one PDF. Returns {pdf, blocks: [...], images: [...], md: str}.
    blocks: {type: text|table|image, text|markdown|img_path, pages: [int]}"""
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    # batch_size=1: avoids RT-DETR "activate padding" tensor error on mixed-size pages
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
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
