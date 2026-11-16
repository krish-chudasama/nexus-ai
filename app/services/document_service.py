"""Document lifecycle: validate -> extract -> chunk -> embed -> store -> register."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.errors import DocumentNotFound, DuplicateDocument
from app.core.logging import get_logger
from app.embeddings.embedders import Embedder
from app.ingestion.chunking import chunk_document
from app.ingestion.extractors import extract
from app.ingestion.security import validate_upload
from app.schemas.models import DocumentInfo, DocumentList
from app.services.registry import DocumentRegistry
from app.vectorstore.chroma_store import ChromaStore, StoredChunk

log = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaStore,
        registry: DocumentRegistry,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.registry = registry

    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        s = self.settings
        name, ext = validate_upload(filename, data, s.max_upload_mb)
        digest = hashlib.sha256(data).hexdigest()
        existing = self.registry.find_by_hash(digest)
        if existing:
            raise DuplicateDocument(
                f"Duplicate of already indexed document '{existing.filename}' (id {existing.id})."
            )

        extracted = extract(data, ext)  # raises ExtractionError -> surfaced to the caller
        chunks = chunk_document(extracted, s.chunk_size, s.chunk_overlap)
        doc_id = uuid.uuid4().hex[:12]
        embeddings = self.embedder.embed_documents([c.text for c in chunks])

        stored = [
            StoredChunk(
                chunk_id=f"{doc_id}:{c.index}",
                doc_id=doc_id,
                filename=name,
                chunk_index=c.index,
                text=c.text,
                page=c.page,
                embedding=e,
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        self.store.add(stored)
        (s.uploads_dir / f"{doc_id}{ext}").write_bytes(data)

        text = extracted.full_text
        info = DocumentInfo(
            id=doc_id,
            filename=name,
            file_type=ext.lstrip("."),
            size_bytes=len(data),
            sha256=digest,
            num_chunks=len(chunks),
            num_chars=len(text),
            num_pages=extracted.num_pages,
            uploaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            preview=text[:300],
        )
        self.registry.add(info)
        log.info(
            "document_indexed", extra={"doc_id": doc_id, "chunks": len(chunks), "doc_name": name}
        )
        return info

    def list(self) -> DocumentList:
        docs = self.registry.list()
        return DocumentList(documents=docs, total_chunks=sum(d.num_chunks for d in docs))

    def get(self, doc_id: str) -> DocumentInfo:
        doc = self.registry.get(doc_id)
        if not doc:
            raise DocumentNotFound(f"No document with id '{doc_id}'.")
        return doc

    def delete(self, doc_id: str) -> None:
        doc = self.get(doc_id)
        self.store.delete_document(doc_id)
        for f in self.settings.uploads_dir.glob(f"{doc_id}.*"):
            f.unlink(missing_ok=True)
        self.registry.remove(doc.id)
        log.info("document_deleted", extra={"doc_id": doc_id})
