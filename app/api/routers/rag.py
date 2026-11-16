from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key, services_dep
from app.schemas.models import QueryRequest, RAGResponse
from app.services.container import Services

router = APIRouter(prefix="/rag", tags=["rag"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=RAGResponse)
def rag_query(req: QueryRequest, svc: Services = Depends(services_dep)) -> RAGResponse:
    """Grounded answer with citations. Requires a configured LLM."""
    return svc.rag.answer(req)


@router.post("/retrieve", response_model=RAGResponse)
def rag_retrieve(req: QueryRequest, svc: Services = Depends(services_dep)) -> RAGResponse:
    """Retrieval debugging: returns ranked chunks + scores without calling the LLM."""
    chunks, debug = svc.rag.retrieve(req)
    return RAGResponse(
        answer="",
        retrieved=chunks,
        insufficient_context=not chunks,
        debug=debug,
        latency_ms=debug.latency_ms,
    )
