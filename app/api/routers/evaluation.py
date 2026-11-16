from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key, services_dep
from app.core.errors import NexusError, ResourceNotFound, ValidationFailed
from app.core.logging import get_logger
from app.evaluation.runner import EvalRunner, save_report
from app.schemas.models import EvalRunRequest, EvalStatus
from app.services.container import Services

log = get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"], dependencies=[Depends(require_api_key)])

_RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_lock = threading.Lock()
_jobs: dict[str, EvalStatus] = {}


def _results_dir(svc: Services) -> Path:
    return svc.settings.evals_dir / "results"


def _dataset_path(svc: Services, name: str) -> Path:
    if Path(name).name != name or not name.endswith(".jsonl"):
        raise ValidationFailed("dataset must be a .jsonl file name inside evals/datasets.")
    path = svc.settings.evals_dir / "datasets" / name
    if not path.is_file():
        raise ValidationFailed(f"Dataset '{name}' not found.")
    return path


def _execute(job_id: str, svc: Services, req: EvalRunRequest, dataset: Path) -> None:
    try:
        runner = EvalRunner(svc.settings, embedder=svc.embedder, llm_factory=svc.llm)
        report = runner.run(
            dataset, use_llm=req.use_llm, use_judge=req.use_judge, top_k=req.top_k, run_id=job_id
        )
        save_report(report, _results_dir(svc))
        status = EvalStatus(run_id=job_id, status="completed", summary=report["summary"])
    except NexusError as exc:
        status = EvalStatus(run_id=job_id, status="failed", error=str(exc))
    except Exception as exc:
        log.error("eval_job_crashed", exc_info=True)
        status = EvalStatus(
            run_id=job_id, status="failed", error=f"Evaluation crashed: {type(exc).__name__}: {exc}"
        )
    with _lock:
        _jobs[job_id] = status


@router.get("/datasets")
def list_datasets(svc: Services = Depends(services_dep)) -> list[str]:
    return sorted(p.name for p in (svc.settings.evals_dir / "datasets").glob("*.jsonl"))


@router.post("/run", response_model=EvalStatus, status_code=202)
def run_eval(req: EvalRunRequest, svc: Services = Depends(services_dep)) -> EvalStatus:
    """Start an evaluation in the background; poll GET /eval/runs/{run_id}."""
    dataset = _dataset_path(svc, req.dataset)
    if req.use_llm:
        svc.llm()  # fail fast (503) when no LLM is configured
    with _lock:
        if any(j.status == "running" for j in _jobs.values()):
            raise NexusError("An evaluation is already running.")
        job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + (
            "-llm" if req.use_llm else "-retrieval"
        )
        _jobs[job_id] = EvalStatus(run_id=job_id, status="running")
    threading.Thread(target=_execute, args=(job_id, svc, req, dataset), daemon=True).start()
    return _jobs[job_id]


@router.get("/runs")
def list_runs(svc: Services = Depends(services_dep)) -> list[dict]:
    """Saved evaluation runs (newest first) with their summaries."""
    rows = []
    for f in sorted(_results_dir(svc).glob("*.json"), reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append(
                {k: r[k] for k in ("run_id", "created_at", "mode", "dataset", "summary")}
                | {"config": r.get("config", {})}
            )
        except (ValueError, KeyError):
            continue
    return rows


@router.get("/runs/{run_id}")
def get_run(run_id: str, svc: Services = Depends(services_dep)) -> dict:
    if not _RUN_ID.match(run_id):
        raise ValidationFailed("Invalid run id.")
    with _lock:
        job = _jobs.get(run_id)
    if job and job.status != "completed":
        return job.model_dump()
    path = _results_dir(svc) / f"{run_id}.json"
    if not path.is_file():
        raise ResourceNotFound(f"No evaluation run '{run_id}'.")
    return json.loads(path.read_text(encoding="utf-8"))
