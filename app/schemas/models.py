"""Typed request/response schemas shared by API, services, UI and MCP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- documents


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    num_chunks: int = 0
    num_chars: int = 0
    num_pages: int | None = None
    uploaded_at: str
    status: Literal["indexed", "failed"] = "indexed"
    error: str | None = None
    preview: str = ""


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str = "Document indexed"


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


# ---------------------------------------------------------------- retrieval / rag


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    page: int | None = None
    text: str
    score: float


class Citation(BaseModel):
    n: int
    doc_id: str
    filename: str
    chunk_index: int
    page: int | None = None
    score: float
    snippet: str


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_ids: list[str] | None = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    rewrite_query: bool | None = None
    rerank: bool | None = None


class RetrievalDebug(BaseModel):
    original_query: str
    effective_query: str
    top_k: int
    min_score: float
    candidates: int
    kept: int
    reranked: bool
    latency_ms: float


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    insufficient_context: bool = False
    debug: RetrievalDebug | None = None
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------- agent


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    max_iterations: int | None = Field(default=None, ge=1, le=15)


class AgentStep(BaseModel):
    step: int
    kind: Literal["tool_call", "final", "error"]
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result_preview: str | None = None
    ok: bool = True
    latency_ms: float = 0.0


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    iterations: int = 0
    stopped_reason: Literal["final", "max_iterations", "error"] = "final"
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------- evaluation


class EvalRunRequest(BaseModel):
    dataset: str = "sample.jsonl"
    use_llm: bool = False
    use_judge: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)


class EvalStatus(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    summary: dict[str, Any] | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------- chat / system


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


class ChatResponse(BaseModel):
    answer: str
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm_configured: bool
    llm_model: str
    embedding_backend: str
    documents: int
    chunks: int
