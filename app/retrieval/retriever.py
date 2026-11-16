"""Semantic retrieval with metadata filtering, score threshold and optional rerank."""

from __future__ import annotations

import time

from app.core.config import Settings
from app.embeddings.embedders import Embedder
from app.retrieval.query import normalize_query
from app.retrieval.rerank import get_reranker
from app.schemas.models import RetrievalDebug, RetrievedChunk
from app.vectorstore.chroma_store import ChromaStore

RERANK_OVERFETCH = 4


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder, store: ChromaStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        rerank: bool | None = None,
        min_score: float | None = None,
        original_query: str | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        s = self.settings
        k = top_k or s.top_k
        threshold = s.min_score if min_score is None else min_score
        do_rerank = s.enable_rerank if rerank is None else rerank
        q = normalize_query(query)

        start = time.perf_counter()
        fetch = k * RERANK_OVERFETCH if do_rerank else k
        candidates = self.store.query(self.embedder.embed_query(q), fetch, doc_ids)
        n_candidates = len(candidates)

        if do_rerank and candidates:
            candidates = get_reranker(s).rerank(q, candidates)
        kept = [c for c in candidates if c.score >= threshold][:k]

        debug = RetrievalDebug(
            original_query=original_query or query,
            effective_query=q,
            top_k=k,
            min_score=threshold,
            candidates=n_candidates,
            kept=len(kept),
            reranked=bool(do_rerank and n_candidates),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return kept, debug
