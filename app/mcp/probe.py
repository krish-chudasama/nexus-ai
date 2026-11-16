"""Live MCP health probe + client-config generation (used by the API / UI)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import StdioServerParameters
from mcp.client import Client

from app.core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def server_env(settings: Settings) -> dict[str, str]:
    """Environment the MCP server needs to open the same index as the API."""
    return {
        "DATA_DIR": str(settings.data_dir.resolve()),
        "EMBEDDING_BACKEND": settings.embedding_backend,
        "EMBEDDING_MODEL": settings.embedding_model,
        "MIN_SCORE": str(settings.min_score),
    }


def client_config(settings: Settings) -> dict[str, Any]:
    """A ready-to-paste `mcpServers` entry for desktop MCP clients (stdio transport)."""
    return {
        "mcpServers": {
            "nexus-ai": {
                "command": sys.executable,
                "args": ["-m", "app.mcp.server"],
                "cwd": str(PROJECT_ROOT),
                "env": server_env(settings),
            }
        }
    }


async def _probe(settings: Settings, timeout_s: float) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, **server_env(settings), "LOG_LEVEL": "WARNING"},
    )
    start = time.perf_counter()
    with anyio.fail_after(timeout_s):
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
    return {
        "ok": True,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ],
        "error": None,
    }


def probe_server(settings: Settings, timeout_s: float = 30.0) -> dict[str, Any]:
    """Spawn the MCP server over stdio, run the handshake and list tools."""
    try:
        return anyio.run(_probe, settings, timeout_s)
    except Exception as exc:  # TimeoutError and anyio ExceptionGroup both derive from Exception
        return {
            "ok": False,
            "latency_ms": None,
            "tools": [],
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }
