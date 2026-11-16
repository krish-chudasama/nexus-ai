"""Select an LLM client from configuration."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.llm.base import LLMClient
from app.llm.openai_compat import OpenAICompatibleClient


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """All supported providers speak the OpenAI chat-completions protocol, so switching
    provider/model is purely LLM_BASE_URL / LLM_API_KEY / LLM_MODEL configuration.
    Add a new LLMClient subclass here for non-compatible providers."""
    return OpenAICompatibleClient(settings or get_settings())
