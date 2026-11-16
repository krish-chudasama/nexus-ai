from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key, services_dep
from app.schemas.models import AgentRequest, AgentResponse
from app.services.container import Services

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=AgentResponse)
def agent_query(req: AgentRequest, svc: Services = Depends(services_dep)) -> AgentResponse:
    """Run the tool-calling agent. Returns the answer plus a step-by-step execution trace."""
    return svc.agent.run(req)


@router.get("/tools")
def agent_tools(svc: Services = Depends(services_dep)) -> list[dict]:
    """Tool schemas exposed to the agent (all read-only)."""
    return svc.agent.registry.openai_schemas()
