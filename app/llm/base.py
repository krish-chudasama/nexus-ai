"""Provider-agnostic LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.schemas.models import Usage


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    latency_ms: float = 0.0
    model: str = ""


class LLMClient(ABC):
    """Implementations must raise ConfigurationError / LLMError, never leak SDK exceptions."""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self, messages: list[dict[str, Any]], *, temperature: float | None = None
    ) -> Iterator[str]: ...
