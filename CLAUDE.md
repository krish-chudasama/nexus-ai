# NEXUS AI — working notes for Claude

Python 3.11+ RAG + agent + MCP + evals app. FastAPI backend, Streamlit UI, ChromaDB, OpenAI-compatible LLMs.

## Commands
- `make test` / `make lint` / `make format`
- `make api` (:8000) · `make ui` (:8501) · `make mcp` (stdio)
- `make eval` (retrieval only, no key) · `make eval-llm` (needs `LLM_API_KEY`)

## Architecture rules
- Business logic lives in `app/services`, `app/rag`, `app/retrieval`, `app/agents`, `app/tools`. The UI (`app/frontend`) only talks to the API via `api_client.py`.
- `app/services/container.py` wires everything (`build_services`). Tests build isolated services with a tmp `data_dir`.
- **One tool implementation, two surfaces:** `app/tools/capabilities.py` is shared by the agent (`tools/registry.py`) and the MCP server (`mcp/server.py`). All tools are read-only.
- Errors: raise `NexusError` subclasses (`core/errors.py`); the API maps them to status codes.
- LLM access only through `llm/base.LLMClient`. Tests use `tests/conftest.FakeLLM`; never call a real LLM in tests.
- Embeddings: `sentence-transformers` (default, needs `.[ml]` + model download) or `hashing` (offline, lexical, for tests/CI).

## Verified SDK facts (installed versions)
- `mcp` 2.x: `from mcp.server.mcpserver import MCPServer` (NOT `FastMCP`); client is `from mcp.client import Client` (async ctx manager, `call_tool` result has `.is_error`, `.structured_content`).
- Streamlit 1.67: `st.Page` with callables, `width="stretch"`.
- PyMuPDF: `import pymupdf` (not `fitz`).

## Security invariants (do not weaken)
- Retrieved text is untrusted: delimiters neutralised in `rag/context.py`, system prompts say so, agent can only call allow-listed tools, tool args validated by pydantic.
- Calculator parses with `ast` allow-list; no `eval`.
- Uploads: extension allow-list, magic-byte check, size cap, `safe_filename`.
- Secrets are `SecretStr`, masked in `/config`, redacted in logs. MCP stdio logs go to stderr only.

## Status
See `IMPLEMENTATION_STATUS.md` for verified results and known gaps.
