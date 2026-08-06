"""Structure-aware chunking of parsed PDF blocks into retrieval chunks.

Design (from the plan): tables and images are their own chunks (they are dense
signal units); runs of text blocks are grouped into ~child_tokens chunks with a
small overlap; every chunk keeps {doc_id, page, type, source} so the demo can
cite page numbers and the eval can map page→chunk.

Token estimate: Chinese is roughly ~0.7-1.0 token/char for bge-family
tokenizers; a conservative constant keeps chunks under the encoder's 512-token
window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 1.4  # 1 token ≈ 1.4 Chinese chars (≈0.7 token/char)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page: int
    ctype: str  # text | table | image
    img_path: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "doc_id": self.doc_id, "text": self.text,
                "page": self.page, "type": self.ctype, "img_path": self.img_path, "meta": self.meta}


def _chars(text: str) -> int:
    return len(text)


def chunk_document(doc_id: str, blocks: list[dict], child_tokens: int = 256,
                   overlap_tokens: int = 32, min_chars: int = 16) -> list[Chunk]:
    """blocks: list of {type, text, pages}. Returns chunks for one document."""
    chunks: list[Chunk] = []
    text_buf: list[str] = []
    text_pages: set[int] = set()

    def flush_text():
        nonlocal text_buf, text_pages
        if not text_buf:
            return
        text = "\n".join(text_buf)
        # split oversized runs by sliding window
        step = max(1, child_tokens - overlap_tokens) * CHARS_PER_TOKEN
        start = 0
        while start < len(text):
            seg = text[start: start + child_tokens * CHARS_PER_TOKEN]
            if len(seg) < min_chars:
                break
            chunks.append(Chunk(
                id=f"{doc_id}_t{len(chunks)}", doc_id=doc_id, text=seg,
                page=sorted(text_pages)[0], ctype="text",
            ))
            start += int(step)
        text_buf = []
        text_pages = set()

    for b in blocks:
        btype, text, pages = b.get("type"), b.get("text", ""), b.get("pages", [0])
        page = sorted(pages)[0] if pages else 0
        if btype == "text":
            if text.strip():
                text_buf.append(text.strip())
                text_pages.add(page)
        else:  # table or image: flush text run, then emit its own chunk
            flush_text()
            if text.strip():
                chunks.append(Chunk(
                    id=f"{doc_id}_t{len(chunks)}", doc_id=doc_id, text=text.strip(),
                    page=page, ctype=btype, img_path=b.get("img_path"),
                ))
    flush_text()
    return chunks
