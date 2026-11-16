from __future__ import annotations

import streamlit as st

from app.frontend import nav
from app.frontend.state import safe_health
from app.frontend.theme import card, page_header, pill


def render() -> None:
    page_header(
        "NEXUS AI",
        "Ask questions about your documents and get grounded answers with citations, "
        "or let an agent use tools on your behalf.",
    )
    health = safe_health()
    if health is None:
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", health["documents"])
    c2.metric("Indexed chunks", health["chunks"])
    c3.metric("Embeddings", health["embedding_backend"])
    c4.metric("LLM model", health["llm_model"])

    status = (
        pill("LLM configured", "ok")
        if health["llm_configured"]
        else pill("LLM key missing", "warn")
    )
    st.markdown(f"{pill('API online', 'ok')} &nbsp; {status}", unsafe_allow_html=True)
    if not health["llm_configured"]:
        st.info(
            "Retrieval and document indexing work without an LLM, but answers need an API key. "
            "Add `LLM_API_KEY` to `.env` (see `.env.example`) and restart the API.",
            icon=":material/key:",
        )
    if health["embedding_backend"].startswith("hashing"):
        st.info(
            "Running on the offline lexical fallback embedder. For semantic search install the "
            "`ml` extra and set `EMBEDDING_BACKEND=sentence-transformers`.",
            icon=":material/info:",
        )

    st.subheader("Get started")
    a, b, c = st.columns(3)
    with a:
        card(
            "1 · Add documents",
            "Upload PDF, DOCX, TXT or Markdown files to build your knowledge base.",
        )
        st.page_link(nav.PAGES["documents"], label="Open library", icon=":material/folder:")
    with b:
        card(
            "2 · Ask questions", "Chat directly, ask with RAG for cited answers, or use the agent."
        )
        st.page_link(nav.PAGES["chat"], label="Open chat", icon=":material/chat:")
    with c:
        card("3 · Measure quality", "Run the evaluation suite to see retrieval and answer metrics.")
        st.page_link(nav.PAGES["evaluation"], label="Open evaluation", icon=":material/analytics:")

    st.subheader("What's inside")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        card("RAG", "Semantic retrieval over ChromaDB with scores, filters and optional reranking.")
    with r2:
        card(
            "Agent", "A bounded tool-calling loop: search, metadata, listing and a safe calculator."
        )
    with r3:
        card(
            "MCP server",
            "The same tools exposed over the Model Context Protocol for other clients.",
        )
    with r4:
        card("Evals", "JSONL datasets with deterministic retrieval, citation and answer metrics.")
