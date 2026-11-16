"""Verifies the MCP server end-to-end: spawns it as a subprocess over stdio and talks to it
with the official MCP client. Proves the tools are real MCP tools, not just Python functions."""

from __future__ import annotations

import json
import os
import sys

import pytest
from mcp import StdioServerParameters
from mcp.client import Client

from app.core.config import Settings
from app.embeddings.embedders import HashingEmbedder
from app.services.container import build_services
from tests.conftest import CORPUS

EXPECTED_TOOLS = {"search_documents", "get_document_info", "list_documents", "calculate"}


@pytest.fixture
def server_params(tmp_path):
    data_dir = tmp_path / "data"
    settings = Settings(
        _env_file=None, data_dir=data_dir, embedding_backend="hashing", min_score=0.05
    )
    svc = build_services(settings, HashingEmbedder())
    for f in sorted(CORPUS.glob("*.md")):
        svc.documents.ingest(f.name, f.read_bytes())
    env = {
        **os.environ,
        "DATA_DIR": str(data_dir),
        "EMBEDDING_BACKEND": "hashing",
        "MIN_SCORE": "0.05",
        "LOG_LEVEL": "WARNING",
    }
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.server"], env=env)


def _payload(result) -> dict:
    """Structured content if provided, else parse the JSON text block."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_lists_exactly_the_expected_tools_with_schemas(server_params):
    async with Client(server_params) as client:
        tools = (await client.list_tools()).tools
        by_name = {t.name: t for t in tools}
        assert set(by_name) == EXPECTED_TOOLS
        schema = by_name["search_documents"].input_schema
        assert schema["required"] == ["query"]
        assert schema["properties"]["top_k"]["maximum"] == 10
        assert by_name["calculate"].input_schema["properties"]["expression"]["maxLength"] == 200
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)


async def test_tools_execute_against_real_index(server_params):
    async with Client(server_params) as client:
        listing = _payload(await client.call_tool("list_documents", {}))
        assert listing["count"] == 3
        names = {d["filename"] for d in listing["documents"]}
        assert names == {"remote_work_policy.md", "product_faq.md", "incident_response.md"}

        hits = _payload(
            await client.call_tool(
                "search_documents", {"query": "Pro plan API requests per minute"}
            )
        )
        assert hits["count"] >= 1 and hits["results"][0]["filename"] == "product_faq.md"

        info = _payload(await client.call_tool("get_document_info", {"doc_id": "product_faq.md"}))
        assert info["filename"] == "product_faq.md" and info["num_chunks"] >= 1

        calc = _payload(await client.call_tool("calculate", {"expression": "12*29 + sqrt(16)"}))
        assert calc["result"] == 352


async def test_errors_are_reported_as_tool_errors(server_params):
    async with Client(server_params) as client:
        bad_doc = await client.call_tool("get_document_info", {"doc_id": "nope.pdf"})
        assert bad_doc.is_error
        bad_expr = await client.call_tool(
            "calculate", {"expression": "__import__('os').system('id')"}
        )
        assert bad_expr.is_error
        invalid = await client.call_tool("search_documents", {"query": "x", "top_k": 99})
        assert invalid.is_error  # schema validation (max 10)
        missing = await client.call_tool("search_documents", {})
        assert missing.is_error


def test_probe_and_client_config(settings):
    from app.mcp.probe import client_config, probe_server

    status = probe_server(settings)
    assert status["ok"] and {t["name"] for t in status["tools"]} == EXPECTED_TOOLS
    cfg = client_config(settings)["mcpServers"]["nexus-ai"]
    assert cfg["args"] == ["-m", "app.mcp.server"] and cfg["env"]["DATA_DIR"]
