from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import require_api_key, services_dep
from app.core.errors import FileTooLarge, ValidationFailed
from app.schemas.models import DocumentInfo, DocumentList, UploadResponse
from app.services.container import Services

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...), svc: Services = Depends(services_dep)
) -> UploadResponse:
    limit = svc.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)  # never buffer more than the limit
    if len(data) > limit:
        raise FileTooLarge(f"File exceeds the {svc.settings.max_upload_mb} MB limit.")
    if not file.filename:
        raise ValidationFailed("A filename is required.")
    info = svc.documents.ingest(file.filename, data)  # CPU-bound; sync on purpose below
    return UploadResponse(document=info, message=f"Indexed {info.num_chunks} chunks")


@router.get("", response_model=DocumentList)
def list_documents(svc: Services = Depends(services_dep)) -> DocumentList:
    return svc.documents.list()


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, svc: Services = Depends(services_dep)) -> DocumentInfo:
    return svc.documents.get(doc_id)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, svc: Services = Depends(services_dep)) -> None:
    svc.documents.delete(doc_id)
