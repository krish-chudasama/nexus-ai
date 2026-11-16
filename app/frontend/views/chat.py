from __future__ import annotations

import html
import time

import streamlit as st

from app.frontend.api_client import ApiError
from app.frontend.state import get_client, safe_health, show_error
from app.frontend.theme import page_header

MODES = {
    "Chat": "Plain conversation with the LLM (no documents).",
    "RAG": "Answers grounded in your documents, with citations.",
    "Agent": "An agent that decides when to search, look up metadata or calculate.",
}
HISTORY_TURNS = 12


def _history() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][
        -HISTORY_TURNS:
    ]


# ------------------------------------------------------------------ rendering


def _caption(meta: dict) -> str:
    bits = [meta["mode"]]
    if meta.get("model"):
        bits.append(meta["model"])
    if meta.get("latency_ms") is not None:
        bits.append(f"{meta['latency_ms'] / 1000:.1f}s")
    usage = meta.get("usage") or {}
    if usage.get("total_tokens"):
        bits.append(f"{usage['total_tokens']} tokens")
    return " · ".join(bits)


def _sources(citations: list[dict]) -> None:
    with st.expander(f"Sources ({len(citations)})", icon=":material/format_quote:"):
        for c in citations:
            page = f" · p.{c['page']}" if c.get("page") else ""
            st.markdown(
                f'<div class="nx-src"><b>[{c["n"]}] {html.escape(c["filename"])}</b>'
                f"<small>{page} · chunk {c['chunk_index']} · score {c['score']:.2f}</small><br>"
                f"{html.escape(c['snippet'])}</div>",
                unsafe_allow_html=True,
            )


def _context_inspector(meta: dict) -> None:
    debug, chunks = meta.get("debug"), meta.get("retrieved") or []
    with st.expander(f"Retrieved context ({len(chunks)} chunks)", icon=":material/search:"):
        if debug:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidates", debug["candidates"])
            c2.metric("Kept", debug["kept"])
            c3.metric("Min score", debug["min_score"])
            c4.metric("Retrieval", f"{debug['latency_ms']:.0f} ms")
            st.caption(
                f"Query used: `{debug['effective_query']}`"
                + (" · reranked" if debug["reranked"] else "")
            )
        for i, c in enumerate(chunks, start=1):
            page = f" · p.{c['page']}" if c.get("page") else ""
            st.markdown(f"**[{i}] {c['filename']}**{page} · score `{c['score']:.3f}`")
            st.code(c["text"], language=None, wrap_lines=True)


def _trace(steps: list[dict]) -> None:
    with st.expander(f"Agent trace ({len(steps)} steps)", icon=":material/account_tree:"):
        for s in steps:
            if s["kind"] == "tool_call":
                badge = "✅" if s["ok"] else "⚠️"
                st.markdown(
                    f'<div class="nx-step">{badge} '
                    f"<b>Step {s['step']} · {html.escape(s['tool'])}</b>"
                    f" <small>({s['latency_ms']:.0f} ms)</small></div>",
                    unsafe_allow_html=True,
                )
                st.json(s["arguments"] or {}, expanded=False)
                st.caption(s["result_preview"] or "")
            else:
                st.markdown(
                    f'<div class="nx-step">🏁 <b>Step {s["step"]} · final answer</b></div>',
                    unsafe_allow_html=True,
                )


def _render_meta(meta: dict | None) -> None:
    if not meta:
        return
    st.caption(_caption(meta))
    if meta.get("citations"):
        _sources(meta["citations"])
    if meta.get("retrieved") or meta.get("debug"):
        _context_inspector(meta)
    if meta.get("steps"):
        _trace(meta["steps"])


# ------------------------------------------------------------------ calls


def _ask(mode: str, prompt: str, opts: dict) -> tuple[str, dict]:
    client, history = get_client(), _history()
    started = time.perf_counter()
    if mode == "Chat":
        text = st.write_stream(client.chat_stream(prompt, history))
        return text, {"mode": mode, "latency_ms": (time.perf_counter() - started) * 1000}
    if mode == "RAG":
        r = client.rag(
            {
                "question": prompt,
                "history": history,
                "top_k": opts["top_k"],
                "rerank": opts["rerank"],
                "rewrite_query": opts["rewrite"],
                "doc_ids": opts["doc_ids"] or None,
            }
        )
        st.markdown(r["answer"])
        return r["answer"], {
            "mode": mode,
            "model": r.get("model"),
            "latency_ms": r["latency_ms"],
            "usage": r.get("usage"),
            "citations": r["citations"],
            "retrieved": r["retrieved"],
            "debug": r.get("debug"),
        }
    r = client.agent(
        {"question": prompt, "history": history, "max_iterations": opts["max_iterations"]}
    )
    st.markdown(r["answer"])
    return r["answer"], {
        "mode": mode,
        "model": r.get("model"),
        "latency_ms": r["latency_ms"],
        "usage": r.get("usage"),
        "citations": r["citations"],
        "steps": r["steps"],
    }


def _options(mode: str) -> dict:
    opts = {"top_k": 5, "rerank": False, "rewrite": False, "doc_ids": [], "max_iterations": 5}
    if mode == "Chat":
        return opts
    with st.expander("Options", icon=":material/tune:"):
        if mode == "RAG":
            c1, c2, c3 = st.columns(3)
            opts["top_k"] = c1.slider("Passages (top-k)", 1, 15, 5)
            opts["rerank"] = c2.toggle("Rerank results", help="Slower; can improve precision.")
            opts["rewrite"] = c3.toggle("Rewrite query", help="Uses one extra LLM call.")
            try:
                docs = get_client().documents()["documents"]
            except ApiError:
                docs = []
            names = {d["id"]: d["filename"] for d in docs}
            opts["doc_ids"] = st.multiselect(
                "Restrict to documents (optional)", list(names), format_func=names.get
            )
        else:
            opts["max_iterations"] = st.slider("Max tool iterations", 1, 10, 5)
    return opts


def render() -> None:
    page_header("Chat", "Talk to the model, query your documents, or hand a task to the agent.")
    health = safe_health()
    if health is None:
        st.stop()
    st.session_state.setdefault("messages", [])

    top, clear = st.columns([5, 1])
    with top:
        mode = st.segmented_control(
            "Mode",
            list(MODES),
            default="RAG",
            required=True,
            label_visibility="collapsed",
            key="chat_mode",
        )
        st.caption(MODES[mode])
    with clear:
        if st.button("Clear", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if mode != "Chat" and health["documents"] == 0:
        st.info(
            "Your library is empty. Upload documents first so there is something to search.",
            icon=":material/folder_open:",
        )
    if not health["llm_configured"]:
        st.warning(
            "No LLM key configured: set `LLM_API_KEY` in `.env` and restart the API.",
            icon=":material/key:",
        )

    opts = _options(mode)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            _render_meta(m.get("meta"))

    if prompt := st.chat_input("Ask something…"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…") if mode != "Chat" else st.container():
                    text, meta = _ask(mode, prompt, opts)
            except ApiError as exc:
                show_error(exc)
                return
            _render_meta(meta)
        st.session_state.messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text, "meta": meta},
        ]
