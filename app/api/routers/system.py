from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key, services_dep
from app.schemas.models import HealthResponse
from app.services.container import Services

APP_VERSION = "0.1.0"
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(svc: Services = Depends(services_dep)) -> HealthResponse:
    docs = svc.documents.list()
    return HealthResponse(
        version=APP_VERSION,
        llm_configured=svc.settings.llm_configured,
        llm_model=svc.settings.llm_model,
        embedding_backend=svc.embedder.name,
        documents=len(docs.documents),
        chunks=docs.total_chunks,
    )


@router.get("/config", dependencies=[Depends(require_api_key)])
def public_config(svc: Services = Depends(services_dep)) -> dict:
    """Effective settings with secrets masked."""
    return svc.settings.public_dict()
