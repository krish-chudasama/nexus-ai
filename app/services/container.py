"""Lazy application container: one place that wires settings, embedder, store, services."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.embeddings.embedders import Embedder, build_embedder
from app.llm.base import LLMClient
from app.llm.factory import get_llm_client
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.services.document_service import DocumentService
from app.services.registry import DocumentRegistry
from app.vectorstore.chroma_store import ChromaStore


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    store: ChromaStore
    registry: DocumentRegistry
    documents: DocumentService
    retriever: Retriever
    rag: RAGPipeline
    agent: Any = None  # app.agents.agent.Agent (set in build_services)

    def llm(self) -> LLMClient:
        """Built per call so a key added to .env / settings is picked up without restart logic."""
        return get_llm_client(self.settings)


def build_services(settings: Settings | None = None, embedder: Embedder | None = None) -> Services:
    settings = settings or get_settings()
    settings.ensure_dirs()
    embedder = embedder or build_embedder(settings)
    store = ChromaStore(settings.chroma_dir)
    registry = DocumentRegistry(settings.registry_path)
    documents = DocumentService(settings, embedder, store, registry)
    retriever = Retriever(settings, embedder, store)
    svc = Services(settings, embedder, store, registry, documents, retriever, None)  # type: ignore[arg-type]
    svc.rag = RAGPipeline(settings, retriever, svc.llm)
    from app.agents.agent import Agent  # local import avoids a circular import with tools

    svc.agent = Agent(settings, svc, svc.llm)
    return svc


_lock = threading.Lock()
_instance: Services | None = None


def get_services() -> Services:
    global _instance
    with _lock:
        if _instance is None:
            _instance = build_services()
        return _instance


def reset_services() -> None:
    global _instance
    with _lock:
        _instance = None
