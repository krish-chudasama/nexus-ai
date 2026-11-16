"""API tests via TestClient with an isolated data dir and the offline embedder."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import create_app
from app.core.config import Settings
from app.embeddings.embedders import HashingEmbedder
from app.services.container import build_services
from tests.conftest import CORPUS, FakeLLM


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_backend="hashing",
        llm_api_key="",
        min_score=0.05,
        max_upload_mb=1,
    )
    svc = build_services(settings, HashingEmbedder())
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    app = create_app()
    app.dependency_overrides[deps.services_dep] = lambda: svc
    return TestClient(app), svc


def _upload(client, name="product_faq.md"):
    return client.post("/documents", files={"file": (name, (CORPUS / name).read_bytes())})


def test_health(api):
    client, _ = api
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm_configured"] is False


def test_upload_list_delete_flow(api):
    client, _ = api
    r = _upload(client)
    assert r.status_code == 201
    doc = r.json()["document"]
    assert doc["num_chunks"] > 0
    assert client.get("/documents").json()["total_chunks"] == doc["num_chunks"]
    assert client.get(f"/documents/{doc['id']}").status_code == 200
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404


def test_duplicate_upload_conflict(api):
    client, _ = api
    _upload(client)
    r = _upload(client)
    assert r.status_code == 409 and r.json()["code"] == "duplicate_document"


def test_rejects_bad_type_and_oversize(api):
    client, _ = api
    r = client.post("/documents", files={"file": ("x.exe", b"MZ....")})
    assert r.status_code == 422 and r.json()["code"] == "unsupported_file_type"
    r = client.post("/documents", files={"file": ("big.txt", b"a" * (2 * 1024 * 1024))})
    assert r.status_code == 413


def test_path_traversal_filename_sanitised(api):
    client, _ = api
    r = client.post("/documents", files={"file": ("../../evil.txt", b"harmless text content")})
    assert r.status_code == 201 and r.json()["document"]["filename"] == "evil.txt"


def test_retrieve_works_without_llm(api):
    client, _ = api
    _upload(client)
    r = client.post("/rag/retrieve", json={"question": "API rate limit for the Pro plan"})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"][0]["filename"] == "product_faq.md"
    assert body["debug"]["kept"] >= 1


def test_rag_query_without_key_gives_clear_config_error(api):
    client, _ = api
    _upload(client)
    r = client.post("/rag/query", json={"question": "API rate limit for the Pro plan"})
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["message"]


def test_rag_query_with_mocked_llm(api):
    client, svc = api
    _upload(client)
    svc.rag._llm_factory = lambda: FakeLLM(["Pro allows 600 requests per minute [1]."])
    r = client.post("/rag/query", json={"question": "API rate limit for the Pro plan"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["filename"] == "product_faq.md"


def test_validation_error_on_empty_question(api):
    client, _ = api
    assert client.post("/rag/query", json={"question": ""}).status_code == 422


def test_api_key_enforced(api, monkeypatch):
    client, _ = api
    from pydantic import SecretStr

    s = deps.get_settings()
    monkeypatch.setattr(s, "nexus_api_key", SecretStr("secret"))
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open


def test_config_endpoint_masks_secrets(api):
    client, _ = api
    body = client.get("/config").json()
    assert body["llm_api_key"] in ("set", "not set")
