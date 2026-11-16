from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key, services_dep
from app.mcp.probe import client_config, probe_server
from app.services.container import Services

router = APIRouter(prefix="/mcp", tags=["mcp"], dependencies=[Depends(require_api_key)])


@router.get("/config")
def mcp_config(svc: Services = Depends(services_dep)) -> dict:
    """Transport details and a ready-to-paste MCP client configuration."""
    return {
        "transport": "stdio",
        "run_command": "python -m app.mcp.server",
        "http_command": "python -m app.mcp.server --transport streamable-http --port 8001",
        "client_config": client_config(svc.settings),
    }


@router.get("/status")
def mcp_status(svc: Services = Depends(services_dep)) -> dict:
    """Spawns the MCP server, performs the protocol handshake and lists its tools."""
    return probe_server(svc.settings)
