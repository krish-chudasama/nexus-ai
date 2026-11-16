"""Optional rerankers. Disabled by default (ENABLE_RERANK=false)."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.models import RetrievedChunk

log = get_logger(__name__)
_TOK = re.compile(r"[a-z0-9]+")


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class LexicalReranker(Reranker):
    """Blends vector score with query-term coverage (IDF-weighted within the candidate set).
    Cheap and dependency-free; helps when a candidate contains the exact query terms."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q_terms = set(_TOK.findall(query.lower()))
        if not q_terms or not chunks:
            return chunks
        toks = [set(_TOK.findall(c.text.lower())) for c in chunks]
        n = len(chunks)
        idf = {t: math.log(1 + n / (1 + sum(t in ts for ts in toks))) for t in q_terms}
        total = sum(idf.values()) or 1.0
        out = []
        for c, ts in zip(chunks, toks, strict=True):
            coverage = sum(idf[t] for t in q_terms if t in ts) / total
            out.append(
                c.model_copy(
                    update={"score": round(self.alpha * c.score + (1 - self.alpha) * coverage, 4)}
                )
            )
        return sorted(out, key=lambda c: c.score, reverse=True)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        scores = self._model.predict([(query, c.text) for c in chunks])
        # Sigmoid keeps scores in 0-1 so they remain comparable with similarity scores.
        out = [
            c.model_copy(update={"score": round(1 / (1 + math.exp(-float(s))), 4)})
            for c, s in zip(chunks, scores, strict=True)
        ]
        return sorted(out, key=lambda c: c.score, reverse=True)


_cached: dict[str, Reranker] = {}


def get_reranker(settings: Settings) -> Reranker:
    key = f"{settings.embedding_backend}:{settings.rerank_model}"
    if key not in _cached:
        if settings.embedding_backend == "sentence-transformers":
            try:
                _cached[key] = CrossEncoderReranker(settings.rerank_model)
            except Exception:
                log.warning("cross_encoder_unavailable_using_lexical", exc_info=True)
                _cached[key] = LexicalReranker()
        else:
            _cached[key] = LexicalReranker()
    return _cached[key]
