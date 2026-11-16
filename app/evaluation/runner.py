"""Evaluation runner (independent of the UI).

    python -m app.evaluation.runner                       # retrieval-only (no LLM needed)
    python -m app.evaluation.runner --llm                 # + answer / citation metrics
    python -m app.evaluation.runner --llm --judge         # + optional LLM-as-judge

By default the corpus in evals/datasets/corpus is indexed into a throwaway store, so results
are reproducible and your document library is untouched. Use --existing-index to evaluate the
live library instead (expected_sources must then match your uploaded filenames).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import NexusError
from app.core.logging import get_logger, setup_logging
from app.embeddings.embedders import Embedder, build_embedder
from app.evaluation.dataset import EvalCase, load_dataset
from app.evaluation.judge import judge_answer
from app.evaluation.metrics import (
    citation_metrics,
    keyword_recall,
    mean,
    retrieval_metrics,
    token_f1,
)
from app.ingestion.security import ALLOWED_EXTENSIONS
from app.llm.base import LLMClient
from app.schemas.models import QueryRequest
from app.services.container import Services, build_services

log = get_logger(__name__)
CORRECT_KEYWORD_RECALL = 1.0
JUDGE_PASS_SCORE = 4


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        llm_factory: Callable[[], LLMClient] | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self._llm_factory = llm_factory

    # ------------------------------------------------------------------ setup
    def _llm(self) -> LLMClient:
        if self._llm_factory:
            return self._llm_factory()
        from app.llm.factory import get_llm_client

        return get_llm_client(self.settings)

    def _build_services(self, corpus: Path | None, workdir: Path) -> tuple[Services, int]:
        if corpus is None:  # evaluate the live library
            svc = build_services(self.settings, self.embedder)
            svc.rag._llm_factory = self._llm
            return svc, len(svc.documents.list().documents)
        isolated = self.settings.model_copy(update={"data_dir": workdir})
        svc = build_services(isolated, self.embedder)
        svc.rag._llm_factory = self._llm
        files = sorted(
            f for f in corpus.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        )
        for f in files:
            svc.documents.ingest(f.name, f.read_bytes())
        return svc, len(files)

    # ------------------------------------------------------------------ run
    def run(
        self,
        dataset_path: Path,
        *,
        use_llm: bool = False,
        use_judge: bool = False,
        top_k: int | None = None,
        rerank: bool | None = None,
        corpus: Path | None = None,
        existing_index: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        cases = load_dataset(dataset_path)
        if use_judge and not use_llm:
            raise NexusError("--judge requires --llm")
        if use_llm:
            self._llm()  # fail fast with a clear ConfigurationError if no key
        started = time.perf_counter()
        corpus_dir = (
            None if existing_index else (corpus or self.settings.evals_dir / "datasets" / "corpus")
        )

        with tempfile.TemporaryDirectory(prefix="nexus-eval-", ignore_cleanup_errors=True) as tmp:
            svc = None
            try:
                svc, n_docs = self._build_services(corpus_dir, Path(tmp))
                if n_docs == 0:
                    raise NexusError("No documents are indexed; nothing to evaluate against.")
                results = [self._run_case(svc, c, use_llm, use_judge, top_k, rerank) for c in cases]
                embedder_name = svc.embedder.name
            finally:
                if svc is not None:
                    if hasattr(svc, "store") and hasattr(svc.store, "close"):
                        svc.store.close()
                    del svc
                import gc

                gc.collect()

        summary = self._summarize(results, use_llm, use_judge)
        s = self.settings
        return {
            "run_id": run_id
            or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ("-llm" if use_llm else "-retrieval"),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": dataset_path.name,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()[:12],
            "mode": "llm" if use_llm else "retrieval-only",
            "judge": use_judge,
            "config": {
                "embedding": embedder_name,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "top_k": top_k or s.top_k,
                "min_score": s.min_score,
                "rerank": s.enable_rerank if rerank is None else rerank,
                "llm_model": s.llm_model if use_llm else None,
                "corpus": "live-library" if existing_index else str(corpus_dir),
                "documents_indexed": n_docs,
                "python": platform.python_version(),
            },
            "duration_s": round(time.perf_counter() - started, 2),
            "summary": summary,
            "failures": [f for r in results for f in r["failures"]],
            "cases": results,
        }

    def _run_case(
        self,
        svc: Services,
        case: EvalCase,
        use_llm: bool,
        use_judge: bool,
        top_k: int | None,
        rerank: bool | None,
    ) -> dict[str, Any]:
        req = QueryRequest(question=case.question, top_k=top_k, rerank=rerank)
        out: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "answerable": case.answerable,
        }
        failures: list[dict[str, str]] = []

        def fail(kind: str, detail: str) -> None:
            failures.append({"case": case.id, "type": kind, "detail": detail})

        try:
            if use_llm:
                resp = svc.rag.answer(req)
                chunks, answer = resp.retrieved, resp.answer
                abstained = resp.insufficient_context
                out["latency_ms"] = resp.latency_ms
                out["total_tokens"] = resp.usage.total_tokens if resp.usage else None
            else:
                chunks, debug = svc.rag.retrieve(req)
                resp, answer, abstained = None, "", not chunks
                out["latency_ms"] = debug.latency_ms
        except NexusError as exc:
            out.update(error=str(exc), failures=[])
            fail("error", str(exc))
            out["failures"] = failures
            return out

        files = [c.filename for c in chunks]
        out["retrieved"] = [
            {"file": c.filename, "chunk": c.chunk_index, "score": c.score} for c in chunks
        ]
        out["abstained"] = abstained

        if not case.answerable:
            out["abstention_correct"] = abstained
            if not abstained:
                fail(
                    "should_abstain", f"retrieved {files[:3]}" if not use_llm else "answered anyway"
                )
            out["failures"] = failures
            return out

        ret = retrieval_metrics(files, case.expected_sources)
        context_text = " ".join(c.text for c in chunks)
        out["retrieval"] = ret
        out["context"] = {
            "source_precision": ret["precision"],
            "keyword_coverage": keyword_recall(context_text, case.expected_keywords),
        }
        if not ret["hit"]:
            fail("retrieval_miss", f"expected {case.expected_sources}, got {sorted(set(files))}")
        elif (out["context"]["keyword_coverage"] or 1.0) < 1.0:
            fail(
                "context_missing_keywords", f"keywords {case.expected_keywords} not all in context"
            )

        if use_llm and resp is not None:
            kw = keyword_recall(answer, case.expected_keywords)
            cites = [c.filename for c in resp.citations]
            cm = citation_metrics(cites, case.expected_sources)
            out["answer"] = answer
            out["answer_metrics"] = {
                "keyword_recall": kw,
                "token_f1": round(token_f1(answer, case.expected_answer), 4),
                "correct": (kw >= CORRECT_KEYWORD_RECALL) if kw is not None else None,
            }
            out["citation"] = {**cm, "cited": sorted(set(cites))}
            if abstained:
                fail("wrongly_abstained", "model said context was insufficient")
            elif kw is not None and kw < CORRECT_KEYWORD_RECALL:
                fail(
                    "answer_incorrect",
                    f"missing keywords from {case.expected_keywords}: {answer[:120]!r}",
                )
            if not abstained and not cites:
                fail("no_citations", "answer contained no valid citations")
            elif cm["precision"] is not None and cm["precision"] < 1.0:
                fail(
                    "wrong_citation",
                    f"cited {sorted(set(cites))}, expected {case.expected_sources}",
                )
            if use_judge:
                verdict = judge_answer(self._llm(), case.question, case.expected_answer, answer)
                out["judge"] = verdict
                if verdict and verdict["score"] < JUDGE_PASS_SCORE:
                    fail("judge_low_score", f"{verdict['score']}/5: {verdict['reason']}")
        out["failures"] = failures
        return out

    # ------------------------------------------------------------------ aggregate
    @staticmethod
    def _summarize(results: list[dict], use_llm: bool, use_judge: bool) -> dict[str, Any]:
        answerable = [r for r in results if r.get("answerable") and "retrieval" in r]
        unanswerable = [r for r in results if not r.get("answerable")]
        errors = [r for r in results if "error" in r]
        s: dict[str, Any] = {
            "cases": len(results),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "errors": len(errors),
            "failed_cases": len({f["case"] for r in results for f in r["failures"]}),
            "hit_rate": mean([r["retrieval"]["hit"] for r in answerable]),
            "mrr": mean([r["retrieval"]["mrr"] for r in answerable]),
            "recall": mean([r["retrieval"]["recall"] for r in answerable]),
            "context_source_precision": mean(
                [r["context"]["source_precision"] for r in answerable]
            ),
            "context_keyword_coverage": mean(
                [r["context"]["keyword_coverage"] for r in answerable]
            ),
            "abstention_accuracy": mean(
                [1.0 if r["abstention_correct"] else 0.0 for r in unanswerable]
            ),
            "avg_latency_ms": mean([r.get("latency_ms") for r in results]),
        }
        if use_llm:
            am = [r["answer_metrics"] for r in answerable if "answer_metrics" in r]
            ct = [r["citation"] for r in answerable if "citation" in r]
            s |= {
                "answer_accuracy": mean(
                    [float(a["correct"]) for a in am if a["correct"] is not None]
                ),
                "answer_keyword_recall": mean([a["keyword_recall"] for a in am]),
                "answer_token_f1": mean([a["token_f1"] for a in am]),
                "citation_precision": mean([c["precision"] for c in ct]),
                "citation_recall": mean([c["recall"] for c in ct]),
                "avg_total_tokens": mean([r.get("total_tokens") for r in results]),
            }
            if use_judge:
                scores = [r["judge"]["score"] for r in answerable if r.get("judge")]
                s["judge_mean_score"] = mean([float(x) for x in scores])
                s["judge_scored"] = len(scores)
        return s


# ---------------------------------------------------------------------- persistence / CLI


def save_report(report: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def format_summary(report: dict) -> str:
    s, c = report["summary"], report["config"]
    lines = [
        f"NEXUS eval  {report['run_id']}  mode={report['mode']}  dataset={report['dataset']}",
        f"embedding={c['embedding']}  top_k={c['top_k']}  min_score={c['min_score']}  "
        f"rerank={c['rerank']}  docs={c['documents_indexed']}",
        "",
    ]
    for key, val in s.items():
        shown = "n/a" if val is None else (f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append(f"  {key:<26}{shown}")
    if report["failures"]:
        lines += ["", f"Failures ({len(report['failures'])}):"]
        lines += [f"  [{f['case']}] {f['type']}: {f['detail']}" for f in report["failures"][:15]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run NEXUS AI evaluations")
    ap.add_argument("--dataset", default="sample.jsonl", help="file in evals/datasets or a path")
    ap.add_argument("--llm", action="store_true", help="generate answers (needs LLM_API_KEY)")
    ap.add_argument("--judge", action="store_true", help="add optional LLM-as-judge (needs --llm)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--rerank", action="store_true", help="enable reranking for this run")
    ap.add_argument("--corpus", type=Path, default=None, help="directory of documents to index")
    ap.add_argument("--existing-index", action="store_true", help="evaluate the live library")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    settings = get_settings()
    setup_logging("WARNING", stream=sys.stderr)
    ds = Path(args.dataset)
    if not ds.exists():
        ds = settings.evals_dir / "datasets" / args.dataset
    try:
        report = EvalRunner(settings).run(
            ds,
            use_llm=args.llm,
            use_judge=args.judge,
            top_k=args.top_k,
            rerank=True if args.rerank else None,
            corpus=args.corpus,
            existing_index=args.existing_index,
        )
    except NexusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(report))
    if not args.no_save:
        print(f"\nSaved: {save_report(report, settings.evals_dir / 'results')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
