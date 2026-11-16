"""Agent tool registry: schemas, argument validation, guarded execution.

Security model: the agent can only call tools registered here (an explicit allow-list),
every tool is read-only, arguments are validated against typed models, and results are
size-capped. Unknown tool names are never executed."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import NexusError
from app.core.logging import get_logger
from app.tools import capabilities as cap

log = get_logger(__name__)
MAX_RESULT_CHARS = 6000


# ---- argument models (also the source of the JSON schema sent to the LLM) ----


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Natural-language search query.")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of passages to return.")
    doc_id: str | None = Field(default=None, description="Optionally restrict to one document id.")


class DocMetaArgs(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200, description="Document id or exact filename.")


class NoArgs(BaseModel):
    pass


class CalcArgs(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic expression, e.g. '(12*29)+sqrt(16)'. Supports + - * / // % ** "
        "and sqrt, abs, round, min, max, log, log10, floor, ceil, sin, cos, tan, pi, e.",
    )


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., BaseModel]

    def schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


@dataclass
class ToolResult:
    ok: bool
    content: str  # JSON string (ok) or error message (not ok)
    latency_ms: float


TOOLS: list[AgentTool] = [
    AgentTool(
        "search_knowledge_base",
        "Semantic search over the user's uploaded documents. Use for any question about "
        "document contents. Returns numbered passages; cite them as [source].",
        SearchArgs,
        cap.search_knowledge_base,
    ),
    AgentTool(
        "get_document_metadata",
        "Get metadata (type, size, pages, chunks, upload time, preview) for one document.",
        DocMetaArgs,
        cap.get_document_metadata,
    ),
    AgentTool(
        "list_available_documents",
        "List all documents in the knowledge base with ids and filenames.",
        NoArgs,
        cap.list_available_documents,
    ),
    AgentTool(
        "calculator",
        "Evaluate an arithmetic expression exactly. Use for any non-trivial calculation.",
        CalcArgs,
        cap.calculator,
    ),
]


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None):
        self._tools = {t.name: t for t in (tools or TOOLS)}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], ctx: cap.ToolContext) -> ToolResult:
        start = time.perf_counter()

        def done(ok: bool, content: str) -> ToolResult:
            return ToolResult(ok, content, round((time.perf_counter() - start) * 1000, 1))

        tool = self._tools.get(name)
        if tool is None:
            log.warning("unknown_tool_requested", extra={"tool": name[:80]})
            return done(False, f"Unknown tool '{name[:60]}'. Available: {', '.join(self.names)}.")
        if "__invalid_json__" in arguments:
            return done(False, "Tool arguments were not valid JSON.")
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            return done(False, f"Invalid arguments: {problems}")
        try:
            result = tool.handler(ctx, **args.model_dump())
            text = result.model_dump_json()
        except NexusError as exc:
            return done(False, str(exc))
        except Exception:
            log.error("tool_crashed", exc_info=True, extra={"tool": name})
            return done(False, "The tool failed unexpectedly.")
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "...[truncated]"
        return done(True, text)
