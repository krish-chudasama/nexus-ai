from __future__ import annotations

import streamlit as st

from app.frontend.api_client import ApiError
from app.frontend.state import get_client, safe_health, show_error
from app.frontend.theme import page_header

GROUPS = {
    "Model": ["llm_provider", "llm_base_url", "llm_model", "llm_temperature", "llm_api_key"],
    "Embeddings": ["embedding_backend", "embedding_model"],
    "Ingestion": ["chunk_size", "chunk_overlap", "max_upload_mb"],
    "Retrieval": [
        "top_k", "min_score", "max_context_chars", "enable_query_rewrite", "enable_rerank",
        "rerank_model",
    ],
    "Agent": ["agent_max_iterations"],
    "System": ["nexus_api_key", "log_level", "data_dir"],
}  # fmt: skip


def render() -> None:
    page_header("Settings", "Effective configuration (read-only). Secrets are never displayed.")
    if safe_health() is None:
        st.stop()
    try:
        cfg = get_client().config()
    except ApiError as exc:
        show_error(exc)
        st.stop()

    st.info(
        "Settings come from environment variables / `.env`. Edit `.env` and restart the API to "
        "change them — see `.env.example` for every option.",
        icon=":material/settings:",
    )
    for group, keys in GROUPS.items():
        with st.expander(group, expanded=group in ("Model", "Retrieval")):
            for key in keys:
                left, right = st.columns([2, 3])
                left.markdown(f"`{key.upper()}`")
                right.write(str(cfg.get(key)))
    prompt = cfg.get("system_prompt") or "(default system prompt)"
    with st.expander("System prompt"):
        st.write(prompt)
