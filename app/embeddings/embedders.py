"""Embedding backends behind one interface.

* ``sentence-transformers`` - real semantic embeddings (needs the ``ml`` extra + model download).
* ``hashing`` - deterministic, dependency-free, lexical-only. Intended for tests, offline demos
  and CI. It is NOT a substitute for semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger

log = get_logger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "sentence-transformers is not installed. Run `pip install -e '.[ml]'` "
                "or set EMBEDDING_BACKEND=hashing for an offline lexical fallback."
            ) from exc
        log.info("loading_embedding_model", extra={"model": model_name})
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not load embedding model '{model_name}': {str(exc)[:200]}"
            ) from exc
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return [v.tolist() for v in vecs]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())
