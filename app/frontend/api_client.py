"""Thin HTTP client for the NEXUS API. The UI contains no business logic; it only calls this."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from app.core.config import get_settings


class ApiError(Exception):
    """Carries a message that is safe and helpful to show directly to the user."""

    def __init__(self, message: str, code: str = "error", status: int | None = None):
        super().__init__(message)
        self.code, self.status = code, status


def _raise_for(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        body = resp.json()
        code, msg = body.get("code", "error"), body.get("message") or body.get("detail")
    except ValueError:
        code, msg = "error", None
    if resp.status_code == 401:
        msg = "The API rejected the request: set NEXUS_API_KEY in .env to match the server."
    raise ApiError(msg or f"Request failed ({resp.status_code}).", code, resp.status_code)


class NexusClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        s = get_settings()
        self.base_url = (base_url or s.api_url).rstrip("/")
        key = api_key if api_key is not None else s.nexus_api_key.get_secret_value()
        self._headers = {"X-API-Key": key} if key else {}

    def _request(self, method: str, path: str, timeout: float = 60.0, **kw: Any) -> httpx.Response:
        try:
            resp = httpx.request(
                method, f"{self.base_url}{path}", headers=self._headers, timeout=timeout, **kw
            )
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the NEXUS API at {self.base_url}. Start it with `make api`.",
                "unreachable",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                "The request timed out. Try again or use a smaller request.", "timeout"
            ) from exc
        _raise_for(resp)
        return resp

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        return self._request(method, path, **kw).json()

    # ---- system
    def health(self) -> dict:
        return self._json("GET", "/health", timeout=5.0)

    def config(self) -> dict:
        return self._json("GET", "/config", timeout=10.0)

    # ---- documents
    def upload(self, filename: str, data: bytes) -> dict:
        return self._json("POST", "/documents", timeout=300.0, files={"file": (filename, data)})

    def documents(self) -> dict:
        return self._json("GET", "/documents", timeout=15.0)

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}", timeout=30.0)

    # ---- chat / rag / agent
    def chat(self, message: str, history: list[dict]) -> dict:
        return self._json(
            "POST", "/chat", timeout=120.0, json={"message": message, "history": history}
        )

    def chat_stream(self, message: str, history: list[dict]) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/stream",
                headers=self._headers,
                json={"message": message, "history": history},
                timeout=120.0,
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    _raise_for(resp)
                yield from resp.iter_text()
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the NEXUS API at {self.base_url}. Start it with `make api`.",
                "unreachable",
            ) from exc

    def rag(self, payload: dict) -> dict:
        return self._json("POST", "/rag/query", timeout=180.0, json=payload)

    def agent(self, payload: dict) -> dict:
        return self._json("POST", "/agent/query", timeout=300.0, json=payload)

    # ---- evaluation
    def eval_datasets(self) -> list[str]:
        return self._json("GET", "/eval/datasets", timeout=10.0)

    def eval_run(self, payload: dict) -> dict:
        return self._json("POST", "/eval/run", timeout=30.0, json=payload)

    def eval_status(self, run_id: str) -> dict:
        return self._json("GET", f"/eval/runs/{run_id}", timeout=30.0)

    def eval_runs(self) -> list[dict]:
        return self._json("GET", "/eval/runs", timeout=30.0)

    # ---- mcp
    def mcp_config(self) -> dict:
        return self._json("GET", "/mcp/config", timeout=10.0)

    def mcp_status(self) -> dict:
        return self._json("GET", "/mcp/status", timeout=60.0)
