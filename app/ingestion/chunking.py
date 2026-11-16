"""Boundary-aware sliding-window chunker with page tracking."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.extractors import ExtractedDocument

_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass
class TextChunk:
    index: int
    text: str
    page: int | None = None


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Snap the cut back to the best natural boundary in the last 30% of the window.
            floor = start + int(chunk_size * 0.7)
            for sep in _BOUNDARIES:
                cut = text.rfind(sep, floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return chunks


def chunk_document(doc: ExtractedDocument, chunk_size: int, overlap: int) -> list[TextChunk]:
    out: list[TextChunk] = []
    for seg in doc.segments:
        for piece in split_text(seg.text, chunk_size, overlap):
            out.append(TextChunk(index=len(out), text=piece, page=seg.page))
    return out
