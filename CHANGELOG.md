# Changelog — NEXUS AI

All notable changes and engineering fixes applied to NEXUS AI are documented in this file.

## [0.1.1] - 2026-11-16

### Fixed
- **ChromaDB Windows File Lock & Cleanup Crash (`BUG-01`):**
  - Added `close()` method to `ChromaStore` (`app/vectorstore/chroma_store.py`) to stop Chroma subsystems, close connections, and clear collection references followed by garbage collection.
  - Updated `EvalRunner.run` (`app/evaluation/runner.py`) to wrap temporary directory execution in a `try...finally` block that guarantees `store.close()` is invoked, and configured `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)`.
  - Added teardown to pytest fixtures in `tests/conftest.py`, `tests/integration/test_api.py`, and `tests/integration/test_eval_api.py`.
- **Event Loop Blocking in Ingestion API (`BUG-02`):**
  - Updated `upload_document` in `app/api/routers/documents.py` to run CPU and disk-bound `svc.documents.ingest` via `anyio.to_thread.run_sync`, preventing event loop starvation.
- **Cross-Platform Makefile Compatibility (`BUG-03`):**
  - Added automatic OS detection (`ifeq ($(OS),Windows_NT)`) in `Makefile` to set `PY ?= .venv/Scripts/python.exe` and `PYTHON ?= python` on Windows.
- **Parent Directory Creation in DocumentRegistry (`BUG-04`):**
  - Added `self._path.parent.mkdir(parents=True, exist_ok=True)` in `DocumentRegistry._save()` (`app/services/registry.py`) to prevent `FileNotFoundError` when saving to new paths.
- **Evaluation Job Crash Logging (`BUG-05`):**
  - Enhanced background job error handling in `app/api/routers/evaluation.py` to preserve exception types and messages.

### Added
- Regression test `test_chroma_store_close_releases_resources` in `tests/unit/test_rag.py`.
- Regression test `test_registry_creates_parent_directory` in `tests/unit/test_ingestion.py`.
- Full project audit report `PROJECT_AUDIT.md`.
- Final audit summary and verification report `FINAL_PROJECT_AUDIT.md`.

### Verified
- Automated test suite: 115/115 tests passing.
- Code style: Ruff lint and format check 100% clean across 80 files.
- Live servers: FastAPI (`:8000`), Streamlit (`:8501`), and MCP stdio subprocess probe verified operational.
