# Implementation Status

Last verified: 2026-11-14 (sandbox: Python 3.13, no real LLM key, Hugging Face blocked, no Docker daemon).

## Phases

- [x] **1 Foundation** — config (pydantic-settings, `SecretStr`), JSON logging with redaction, error types, schemas, LLM abstraction, `.env.example`, `pyproject.toml`
- [x] **2 Core LLM + RAG** — ingestion (PDF/DOCX/TXT/MD), chunking, embeddings, Chroma store, retriever (filter/top-k/threshold/rewrite/rerank), grounded generation + citations, FastAPI, Streamlit UI
- [x] **3 Agent + MCP** — tool registry, bounded agent loop with trace, real MCP server (official SDK), MCP probe + client config
- [x] **4 Evaluation** — JSONL dataset, deterministic metrics, optional judge, runner/CLI, eval API (background run + status), UI dashboard
- [x] **5 Finalization** — UI pages (dashboard, documents, chat, evaluation, MCP, settings), Makefile, Dockerfile, compose, README, CLAUDE.md

## Verified (actually run)

| Check | Result |
|---|---|
| `pytest` (unit 87 + integration 26) | **113 passed** |
| `ruff check` / `ruff format --check` | clean (80 files) |
| API launched via `make api`; upload, retrieve, duplicate (409), bad type (422), oversize (413), traversal filename sanitised, API-key auth | works (curl + integration tests) |
| MCP server over stdio, spawned as subprocess, driven by the SDK client: tool list/schemas, all 4 tools executed, error paths (`is_error`) | works (3 tests + live probe via `/mcp/status`) |
| MCP server and API sharing one data dir across processes (upload via API visible to a running MCP session) | works |
| OpenAI-compatible client (real `openai` SDK) vs local stub speaking the wire protocol: auth header, usage, tool calls, malformed tool JSON, streaming, error mapping, full agent loop | works (7 tests) |
| Agent: direct answer, search→cited answer, calculator, max-iteration cap, tool failure, per-turn tool cap, prompt-injection document cannot trigger a non-allow-listed tool | works (scripted LLM) |
| Calculator rejects code-execution / DoS expressions | 24 parametrised cases |
| Evaluation CLI + saved JSON report; `--llm` without key, missing dataset, `--judge` without `--llm` give clear errors (exit 2) | works |
| Evaluation API: background run, polling, listing, path-traversal rejection | works (4 tests) |
| Streamlit UI driven with headless Chromium: every page loads without exceptions; real upload; duplicate shows friendly error; chat without key shows config message; eval run from UI; MCP check from UI; settings hide the key; no horizontal overflow at 390px | works |
| Fresh venv install of `pip install ".[ui]"` (Dockerfile-equivalent, no ML extra) runs the API | works |
| `docker compose config` | valid |

### Baseline retrieval evaluation (toy data — not an accuracy claim)

`make eval` with `EMBEDDING_BACKEND=hashing MIN_SCORE=0.05`, 3 docs / 10 questions: hit rate 1.00, MRR 1.00, recall 1.00, context source precision 0.47, **abstention accuracy 0.00** (the lexical embedder cannot reject the off-topic question; reported as failure `q10 should_abstain`). Answer and citation metrics are `n/a` because no LLM ran.

## NOT verified (needs credentials / network / tooling)

- **Any real LLM call.** No API key here. RAG answers, agent behaviour, query rewriting, LLM-as-judge and `make eval-llm` are verified only against scripted fakes and the local stub server — never a real model.
- **Semantic embeddings.** `sentence-transformers`/PyTorch installed, but the model download (`all-MiniLM-L6-v2`) was blocked (HTTP 403) in this sandbox. The failure path gives a clear `ConfigurationError`; semantic retrieval quality, the cross-encoder reranker, and the default `MIN_SCORE=0.20` threshold are untested. All retrieval numbers above come from the lexical `hashing` backend.
- **Docker image build / `docker compose up`.** No Docker daemon available. Compose file syntax and the install steps (via fresh venv) were checked; the image itself was not built.
- **MCP with a real desktop client** (e.g. a GUI app consuming the generated config) and the streamable-HTTP transport. Only stdio via the SDK client was exercised.
- Scanned PDFs (unsupported by design); large-file/performance behaviour.

## Known issues / notes

- With the `hashing` backend, set `MIN_SCORE=0.05` (documented in README); the default 0.20 filters most results.
- Ingestion runs synchronously inside the request.
- Streaming exists for plain chat only.
- Evaluation answer correctness is keyword-based by design (deterministic); paraphrases can be mis-scored.

## Next actions

1. Add `LLM_API_KEY` to `.env`, run `make eval-llm` and record the real numbers here.
2. Run `make install-ml` with Hugging Face access, set `EMBEDDING_BACKEND=sentence-transformers`, re-run `make eval`, and tune `MIN_SCORE`.
3. `docker compose up --build` on a machine with Docker.
