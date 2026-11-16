import shutil
import time

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import create_app
from app.core.config import Settings
from app.embeddings.embedders import HashingEmbedder
from app.services.container import build_services
from tests.conftest import CORPUS


@pytest.fixture
def client(tmp_path):
    evals = tmp_path / "evals"
    shutil.copytree(CORPUS.parent, evals / "datasets")
    (evals / "results").mkdir()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        evals_dir=evals,
        embedding_backend="hashing",
        llm_api_key="",
        min_score=0.05,
    )
    svc = build_services(settings, HashingEmbedder())
    app = create_app()
    app.dependency_overrides[deps.services_dep] = lambda: svc
    return TestClient(app)


def _wait(client, run_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/eval/runs/{run_id}").json()
        if body.get("status") not in ("running",):
            return body
        time.sleep(0.1)
    raise AssertionError("evaluation did not finish")


def test_run_poll_and_list(client):
    assert "sample.jsonl" in client.get("/eval/datasets").json()
    r = client.post("/eval/run", json={"dataset": "sample.jsonl"})
    assert r.status_code == 202 and r.json()["status"] == "running"
    run_id = r.json()["run_id"]
    report = _wait(client, run_id)
    assert report["mode"] == "retrieval-only" and report["summary"]["cases"] == 10
    listed = client.get("/eval/runs").json()
    assert listed and listed[0]["run_id"] == run_id


def test_llm_eval_without_key_is_clear_error(client):
    r = client.post("/eval/run", json={"dataset": "sample.jsonl", "use_llm": True})
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["message"]


def test_dataset_path_traversal_rejected(client):
    for bad in ("../secret.jsonl", "/etc/passwd", "sub/dir.jsonl", "sample.txt"):
        assert client.post("/eval/run", json={"dataset": bad}).status_code == 422


def test_run_id_validation_and_not_found(client):
    assert client.get("/eval/runs/..%2f..%2fetc").status_code in (404, 422)
    assert client.get("/eval/runs/does-not-exist").status_code == 404
