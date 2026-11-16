"""Plain LLM chat (no retrieval), with optional streaming."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import require_api_key, services_dep
from app.core.errors import LLMError
from app.core.logging import get_logger
from app.schemas.models import ChatRequest, ChatResponse
from app.services.container import Services

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


def _messages(req: ChatRequest, svc: Services) -> list[dict]:
    history = [{"role": m.role, "content": m.content} for m in req.history[-20:]]
    return [
        {"role": "system", "content": svc.settings.effective_system_prompt},
        *history,
        {"role": "user", "content": req.message},
    ]


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, svc: Services = Depends(services_dep)) -> ChatResponse:
    resp = svc.llm().chat(_messages(req, svc))
    return ChatResponse(
        answer=resp.content,
        model=resp.model,
        usage=resp.usage,
        latency_ms=round(resp.latency_ms, 1),
    )


@router.post("/stream")
def chat_stream(req: ChatRequest, svc: Services = Depends(services_dep)) -> StreamingResponse:
    llm = svc.llm()  # raises ConfigurationError before the stream starts -> clean JSON error
    messages = _messages(req, svc)

    def gen() -> Iterator[str]:
        try:
            yield from llm.stream(messages)
        except LLMError as exc:
            log.warning("stream_failed", extra={"error": str(exc)})
            yield f"\n\n[error: {exc}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
