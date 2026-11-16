"""FastAPI dependencies: service access and optional API-key auth."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.core.config import get_settings
from app.services.container import Services, get_services


def services_dep() -> Services:
    return get_services()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If NEXUS_API_KEY is configured, every protected route needs a matching X-API-Key."""
    expected = get_settings().nexus_api_key.get_secret_value()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
