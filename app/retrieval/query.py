"""Query normalisation and optional LLM-based rewriting."""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.llm.base import LLMClient

log = get_logger(__name__)

_REWRITE_PROMPT = (
    "Rewrite the user's latest question as a single standalone search query for a document "
    "search engine. Resolve pronouns using the conversation. Output ONLY the query, no quotes, "
    "no explanation."
)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def rewrite_query(query: str, history: list[dict[str, str]], llm: LLMClient) -> str:
    """Returns a standalone query; falls back to the normalised original on any failure."""
    base = normalize_query(query)
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nQuestion: {base}"},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        rewritten = normalize_query(resp.content.strip().strip('"'))
        return rewritten if 0 < len(rewritten) <= 400 else base
    except Exception:
        log.warning("query_rewrite_failed", exc_info=True)
        return base
