"""ChromaDB persistence layer (cosine space; score = 1 - distance)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.schemas.models import RetrievedChunk

COLLECTION = "nexus_chunks"


@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    page: int | None
    embedding: list[float]


class ChromaStore:
    def __init__(self, path: Path, collection: str = COLLECTION):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[c.embedding for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "page": c.page if c.page is not None else -1,
                }
                for c in chunks
            ],
        )

    def query(
        self, embedding: list[float], k: int, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        total = self._col.count()
        if total == 0:
            return []
        where: dict[str, Any] | None = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            page = meta.get("page", -1)
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    chunk_index=int(meta["chunk_index"]),
                    page=None if page in (-1, None) else int(page),
                    text=doc,
                    score=round(1.0 - float(dist), 4),
                )
            )
        return out

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()

    def count_for(self, doc_id: str) -> int:
        return len(self._col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def close(self) -> None:
        """Release open file handles and stop client subsystems."""
        if hasattr(self, "_client") and self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
                elif hasattr(self._client, "_system") and hasattr(self._client._system, "stop"):
                    self._client._system.stop()
            except Exception:
                pass
            finally:
                self._col = None
                self._client = None
                import gc

                gc.collect()
