.PHONY: install install-ml api ui mcp test lint format eval eval-llm docker-up docker-down

PY ?= .venv/bin/python

install:            ## core + UI + dev deps (offline hashing embedder works without ML extra)
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[ui,dev]"

install-ml:         ## semantic embeddings + cross-encoder reranker (downloads PyTorch)
	$(PY) -m pip install -e ".[ml]"

api:                ## FastAPI backend on :8000
	$(PY) -m uvicorn app.api.main:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

ui:                 ## Streamlit frontend on :8501
	$(PY) -m streamlit run app/frontend/app.py

mcp:                ## MCP server over stdio
	$(PY) -m app.mcp.server

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check app tests && $(PY) -m ruff format --check app tests

format:
	$(PY) -m ruff format app tests && $(PY) -m ruff check --fix app tests

eval:               ## retrieval-only evaluation (no API key needed)
	$(PY) -m app.evaluation.runner

eval-llm:           ## full evaluation incl. answers + citations (needs LLM_API_KEY)
	$(PY) -m app.evaluation.runner --llm

docker-up:
	docker compose up --build

docker-down:
	docker compose down
