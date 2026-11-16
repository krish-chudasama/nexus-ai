"""Shared fixtures. Tests never need network, API keys or model downloads."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.embeddings.embedders import HashingEmbedder
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.services.container import Services, build_services

CORPUS = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "corpus"


class FakeLLM(LLMClient):
    """Scripted LLM. Each call pops the next response; records every request."""

    model = "fake-model"

    def __init__(self, responses: list[LLMResponse | str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        item = self.responses.pop(0) if self.responses else "OK"
        if isinstance(item, str):
            return LLMResponse(content=item, model=self.model)
        return item

    def stream(self, messages, *, temperature=None) -> Iterator[str]:
        self.calls.append({"messages": messages, "stream": True})
        yield from ["Hello", " ", "world"]


def tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        model="fake-model",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_backend="hashing",
        llm_api_key="test-key",
        min_score=0.05,
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def services(settings: Settings) -> Services:
    return build_services(settings, HashingEmbedder())


@pytest.fixture
def indexed(services: Services) -> Services:
    for f in sorted(CORPUS.glob("*.md")):
        services.documents.ingest(f.name, f.read_bytes())
    return services


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
