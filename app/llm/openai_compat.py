"""OpenAI-compatible client (OpenAI, Groq, OpenRouter, Ollama, vLLM, ...)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from app.core.config import Settings
from app.core.errors import ConfigurationError, LLMError
from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.schemas.models import Usage

log = get_logger(__name__)


class OpenAICompatibleClient(LLMClient):
    def __init__(self, settings: Settings):
        key = settings.llm_api_key.get_secret_value().strip()
        if not key:
            raise ConfigurationError(
                "LLM_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        from openai import OpenAI  # imported lazily so tests/MCP work without network config

        self._client = OpenAI(
            base_url=settings.llm_base_url, api_key=key, timeout=settings.llm_timeout_s
        )
        self.model = settings.llm_model
        self._temperature = settings.llm_temperature

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # SDK raises many subclasses; normalise
            raise LLMError(f"LLM request failed: {type(exc).__name__}: {_short(exc)}") from exc
        latency = (time.perf_counter() - start) * 1000

        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except ValueError:
                args = {"__invalid_json__": raw[:200]}
            calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw)
            )
        usage = None
        if resp.usage:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
        log.info("llm_call", extra={"model": self.model, "latency_ms": round(latency, 1)})
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=calls,
            usage=usage,
            latency_ms=latency,
            model=resp.model or self.model,
        )

    def stream(
        self, messages: list[dict[str, Any]], *, temperature: float | None = None
    ) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self._temperature if temperature is None else temperature,
                stream=True,
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except Exception as exc:
            raise LLMError(f"LLM stream failed: {type(exc).__name__}: {_short(exc)}") from exc


def _short(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:300]
