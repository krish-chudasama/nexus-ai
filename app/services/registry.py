"""Thread-safe JSON registry of document metadata (source of truth for the library UI)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from app.schemas.models import DocumentInfo


class DocumentRegistry:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.RLock()
        self._docs: dict[str, DocumentInfo] = {}
        self._mtime_ns = 0
        self._load()

    def _refresh(self) -> None:
        """Reload if another process (API / MCP server) changed the file."""
        try:
            mtime = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            if self._docs:
                self._docs, self._mtime_ns = {}, 0
            return
        if mtime != self._mtime_ns:
            self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        self._mtime_ns = self._path.stat().st_mtime_ns
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._docs = {d["id"]: DocumentInfo(**d) for d in raw}
        except (ValueError, KeyError, TypeError):
            # Corrupt registry: keep a backup and start clean rather than crash the app.
            self._path.rename(self._path.with_suffix(".corrupt.json"))
            self._docs = {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([d.model_dump() for d in self._docs.values()], indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)  # atomic
        self._mtime_ns = self._path.stat().st_mtime_ns

    def add(self, doc: DocumentInfo) -> None:
        with self._lock:
            self._docs[doc.id] = doc
            self._save()

    def get(self, doc_id: str) -> DocumentInfo | None:
        with self._lock:
            self._refresh()
            return self._docs.get(doc_id)

    def find_by_hash(self, sha256: str) -> DocumentInfo | None:
        with self._lock:
            self._refresh()
            return next((d for d in self._docs.values() if d.sha256 == sha256), None)

    def remove(self, doc_id: str) -> DocumentInfo | None:
        with self._lock:
            doc = self._docs.pop(doc_id, None)
            if doc:
                self._save()
            return doc

    def list(self) -> list[DocumentInfo]:
        with self._lock:
            self._refresh()
            return sorted(self._docs.values(), key=lambda d: d.uploaded_at, reverse=True)
