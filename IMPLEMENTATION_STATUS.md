# Implementation Status

Last verified: 2026-11-16 (Host: Windows 11, Python 3.11.8, `.venv`, no external LLM key, Docker daemon offline).

## Phases

- [x] **1 Foundation** — config (pydantic-settings, `SecretStr`), JSON logging with redaction, error types, schemas, LLM abstraction, `.env.example`, `pyproject.toml`
- [x] **2 Core LLM + RAG** — ingestion (PDF/DOCX/TXT/MD), chunking, embeddings (semantic or lexical hashing), Chroma store with resource cleanup, retriever (filter/top-k/threshold/rewrite/rerank), grounded generation + citations, FastAPI (non-blocking async routes), Streamlit UI
- [x] **3 Agent + MCP** — tool registry, bounded agent loop with trace, real MCP server (official SDK), MCP probe + client config
- [x] **4 Evaluation** — JSONL dataset, deterministic metrics, optional judge, runner/CLI (Windows-safe resource release), eval API (background run + polling), UI dashboard
- [x] **5 Finalization** — UI pages (dashboard, documents, chat, evaluation, MCP, settings), cross-platform Makefile, Dockerfile, compose, README, CLAUDE.md

## Verified (actually run)

| Check | Result |
|---|---|
| `pytest` (unit 89 + integration 26) | **115 passed** (100% pass rate) |
| `ruff check` / `ruff format --check` | clean (80 files) |
| Windows SQLite / ChromaDB handle release & temp cleanup | **VERIFIED** (no WinError 32 PermissionError) |
| FastAPI async endpoint non-blocking ingestion | **VERIFIED** (`anyio.to_thread.run_sync`) |
| API launched via uvicorn; health, docs, upload, retrieve, delete | **VERIFIED** (HTTP 200/201/204 on live process) |
| MCP server over stdio, spawned as subprocess, driven by the SDK client | **VERIFIED** (4 tools listed with schemas, probe returned `ok: True`) |
| Evaluation CLI (`python -m app.evaluation.runner`) | **VERIFIED** (runs, calculates metrics, saves JSON) |
| Evaluation API: background run, polling, listing | **VERIFIED** (HTTP 202 -> completed report) |
| Streamlit UI launch | **VERIFIED** (HTTP 200 on `http://localhost:8501`) |
| `docker compose config` | **VERIFIED** (valid syntax and schema) |

### Baseline retrieval evaluation (sample dataset)

`python -m app.evaluation.runner` with `EMBEDDING_BACKEND=hashing MIN_SCORE=0.05`:
- Cases: 10 (9 answerable, 1 unanswerable)
- Hit rate: 1.000
- MRR: 1.000
- Recall: 1.000
- Context source precision: 0.474
- Context keyword coverage: 1.000
- Abstention accuracy: 0.000 (lexical embedder does not reject off-topic query `q10`; reported as expected failure `q10 should_abstain`)

## NOT verified / Environment limitations

- **Real LLM calls:** `LLM_API_KEY` is unconfigured. All LLM endpoints, RAG answer generation, agent tool decisions, and LLM-as-judge are verified with scripted fakes and mock stubs.
- **Docker image build (`docker compose up`):** Docker CLI is present (v29.1.3), but the Docker Desktop daemon is not running on this host. Configuration is statically validated via `docker compose config`.
- **Hugging Face model download:** Semantic embedding model (`all-MiniLM-L6-v2`) requires network download; offline lexical `hashing` backend was used for deterministic local verification.

## Resolved Known Issues

1. **Fixed (P0):** Windows file lock error `WinError 32` when cleaning up temporary ChromaDB directories during evaluation. Resolved by adding `ChromaStore.close()`, explicit collection cleanup, and `TemporaryDirectory(ignore_cleanup_errors=True)`.
2. **Fixed (P1):** Synchronous blocking ingestion in `upload_document` async endpoint. Resolved by offloading to `anyio.to_thread.run_sync`.
3. **Fixed (P1):** Hardcoded Linux paths in `Makefile`. Resolved with OS detection for Windows PowerShell.
4. **Fixed (P2):** Missing directory creation in `DocumentRegistry._save()`. Resolved by calling `mkdir(parents=True, exist_ok=True)`.
