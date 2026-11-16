"""Text extraction for PDF / DOCX / TXT / Markdown, plus cleaning."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from app.core.errors import ExtractionError


@dataclass
class Segment:
    text: str
    page: int | None = None


@dataclass
class ExtractedDocument:
    segments: list[Segment]
    num_pages: int | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    text = _CTRL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> ExtractedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF: {str(exc)[:150]}") from exc
    try:
        if doc.needs_pass:
            raise ExtractionError("PDF is password protected.")
        segs = []
        for i, page in enumerate(doc, start=1):
            txt = clean_text(page.get_text("text"))
            if txt:
                segs.append(Segment(txt, page=i))
        return ExtractedDocument(segs, num_pages=doc.page_count)
    finally:
        doc.close()


def _extract_docx(data: bytes) -> ExtractedDocument:
    import docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {str(exc)[:150]}") from exc
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = clean_text("\n\n".join(parts))
    return ExtractedDocument([Segment(text)] if text else [])


def _extract_text(data: bytes) -> ExtractedDocument:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        raise ExtractionError("Could not decode text file.")
    text = clean_text(text)
    return ExtractedDocument([Segment(text)] if text else [])


def extract(data: bytes, ext: str) -> ExtractedDocument:
    if ext == ".pdf":
        doc = _extract_pdf(data)
    elif ext == ".docx":
        doc = _extract_docx(data)
    else:
        doc = _extract_text(data)
    if not doc.segments:
        raise ExtractionError(
            "No extractable text found (scanned/image-only documents are not supported)."
        )
    return doc
