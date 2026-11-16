from __future__ import annotations

import json

import streamlit as st

from app.frontend.api_client import ApiError
from app.frontend.state import get_client, safe_health, show_error
from app.frontend.theme import page_header, pill


def render() -> None:
    page_header(
        "MCP Server",
        "The same read-only tools exposed over the Model Context Protocol for external clients.",
    )
    if safe_health() is None:
        st.stop()
    client = get_client()

    try:
        cfg = client.mcp_config()
    except ApiError as exc:
        show_error(exc)
        st.stop()

    st.subheader("Status")
    if st.button("Check server", icon=":material/health_and_safety:"):
        with st.spinner("Starting the MCP server and running the protocol handshake…"):
            try:
                st.session_state.mcp_status = client.mcp_status()
            except ApiError as exc:
                show_error(exc)
    status = st.session_state.get("mcp_status")
    if status is None:
        st.caption("Not checked yet. This launches the server over stdio and lists its tools.")
    elif status["ok"]:
        st.markdown(
            f"{pill('Handshake OK', 'ok')} &nbsp; {len(status['tools'])} tools · "
            f"{status['latency_ms']:.0f} ms",
            unsafe_allow_html=True,
        )
        for t in status["tools"]:
            with st.expander(f"`{t['name']}`"):
                st.write(t["description"])
                st.json(t["input_schema"], expanded=False)
    else:
        st.markdown(pill("Server failed to start", "bad"), unsafe_allow_html=True)
        st.error(status["error"])

    st.subheader("Run it")
    st.markdown(
        f"Transport: **{cfg['transport']}** (default). "
        "Stdout carries the protocol; logs go to stderr."
    )
    st.code(cfg["run_command"], language="bash")
    st.caption("Or over HTTP:")
    st.code(cfg["http_command"], language="bash")

    st.subheader("Client configuration")
    st.caption("Paste into a desktop MCP client's config (e.g. `mcpServers` section).")
    st.code(json.dumps(cfg["client_config"], indent=2), language="json")
