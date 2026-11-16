"""Read-only application capabilities. ONE implementation shared by the agent and the MCP
server, so both surfaces behave identically. None of these can modify or delete data."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from app.core.errors import DocumentNotFound
from app.schemas.models import RetrievedChunk
from app.services.container import Services
from app.tools.calculator import calculate


@dataclass
class ToolContext:
    """Per-run state. ``sources`` accumulates every chunk shown to the model so that
    citation numbers ([1], [2]...) stay stable across multiple searches."""

    services: Services
    sources: list[RetrievedChunk] = field(default_factory=list)

    def source_number(self, chunk: RetrievedChunk) -> int:
        for i, existing in enumerate(self.sources, start=1):
            if existing.chunk_id == chunk.chunk_id:
                return i
        self.sources.append(chunk)
        return len(self.sources)


# ------------------------------------------------------------------ result models


class SearchHit(BaseModel):
    source: int
    doc_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    text: str


class SearchResult(BaseModel):
    query: str
    count: int
    results: list[SearchHit]
    note: str = ""


class DocumentSummary(BaseModel):
    id: str
    filename: str
    file_type: str
    num_chunks: int
    num_pages: int | None = None
    uploaded_at: str


class DocumentListResult(BaseModel):
    count: int
    documents: list[DocumentSummary]


class DocumentMetadata(DocumentSummary):
    size_bytes: int
    num_chars: int
    sha256: str
    preview: str


class CalcResult(BaseModel):
    expression: str
    result: float | int


# ------------------------------------------------------------------ capabilities


def search_knowledge_base(
    ctx: ToolContext, query: str, top_k: int = 4, doc_id: str | None = None
) -> SearchResult:
    chunks, _ = ctx.services.retriever.retrieve(
        query, top_k=top_k, doc_ids=[doc_id] if doc_id else None
    )
    hits = [
        SearchHit(
            source=ctx.source_number(c),
            doc_id=c.doc_id,
            filename=c.filename,
            page=c.page,
            chunk_index=c.chunk_index,
            score=c.score,
            text=c.text,
        )
        for c in chunks
    ]
    note = "" if hits else "No sufficiently relevant passages were found."
    return SearchResult(query=query, count=len(hits), results=hits, note=note)


def list_available_documents(ctx: ToolContext) -> DocumentListResult:
    docs = ctx.services.documents.list().documents
    return DocumentListResult(
        count=len(docs),
        documents=[
            DocumentSummary(**d.model_dump(include=set(DocumentSummary.model_fields))) for d in docs
        ],
    )


def get_document_metadata(ctx: ToolContext, doc_id: str) -> DocumentMetadata:
    """Looks up by document id, falling back to an exact (case-insensitive) filename match."""
    registry = ctx.services.registry
    doc = registry.get(doc_id)
    if doc is None:
        wanted = doc_id.strip().lower()
        doc = next((d for d in registry.list() if d.filename.lower() == wanted), None)
    if doc is None:
        raise DocumentNotFound(f"No document with id or filename '{doc_id}'.")
    return DocumentMetadata(**doc.model_dump(include=set(DocumentMetadata.model_fields)))


def calculator(ctx: ToolContext, expression: str) -> CalcResult:  # noqa: ARG001
    return CalcResult(expression=expression, result=calculate(expression))
