"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import agent, chat, documents, evaluation, mcp, rag, system
from app.core.config import get_settings
from app.core.errors import NexusError
from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)

ROUTERS = [
    system.router,
    documents.router,
    rag.router,
    chat.router,
    agent.router,
    evaluation.router,
    mcp.router,
]


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(
        title="NEXUS AI API",
        version=system.APP_VERSION,
        description="RAG, agent and evaluation API for the NEXUS AI knowledge assistant.",
    )

    @app.exception_handler(NexusError)
    async def nexus_error_handler(_: Request, exc: NexusError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("request_failed", extra={"code": exc.code, "error": str(exc)})
        return JSONResponse(
            status_code=exc.status_code, content={"code": exc.code, "message": str(exc)}
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "An unexpected error occurred."},
        )

    for r in ROUTERS:
        app.include_router(r)
    return app


app = create_app()
