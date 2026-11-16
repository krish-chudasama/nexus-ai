"""Context construction (with size limits) and citation extraction."""

from __future__ import annotations

import re

from app.schemas.models import Citation, RetrievedChunk

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _neutralize(text: str) -> str:
    """Prevent chunk text from closing/opening our delimiters (prompt-injection hygiene)."""
    return text.replace("<source", "< source").replace("</source", "< /source")


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> tuple[str, list[RetrievedChunk]]:
    """Packs chunks in rank order until ``max_chars`` is reached.
    Returns (context_text, chunks_actually_included); source ids are 1-based positions."""
    parts: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    for chunk in chunks:
        body = _neutralize(chunk.text)
        page = f' page="{chunk.page}"' if chunk.page else ""
        block = f'<source id="{len(used) + 1}" file="{chunk.filename}"{page}>\n{body}\n</source>'
        if used and total + len(block) > max_chars:
            break
        if not used and len(block) > max_chars:  # always include the best chunk, truncated
            block = block[:max_chars]
        parts.append(block)
        used.append(chunk)
        total += len(block)
    return "\n\n".join(parts), used


def extract_citations(answer: str, used: list[RetrievedChunk]) -> list[Citation]:
    """Only citations that map to a chunk actually shown to the model are returned;
    any other [n] the model invents is ignored (no fabricated citations)."""
    seen: list[int] = []
    for match in _CITE_RE.finditer(answer):
        for num in re.split(r"\s*,\s*", match.group(1)):
            n = int(num)
            if 1 <= n <= len(used) and n not in seen:
                seen.append(n)
    cites = []
    for n in seen:
        c = used[n - 1]
        snippet = c.text[:240] + ("…" if len(c.text) > 240 else "")
        cites.append(
            Citation(
                n=n,
                doc_id=c.doc_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                page=c.page,
                score=c.score,
                snippet=snippet,
            )
        )
    return cites
