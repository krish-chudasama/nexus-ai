"""Streamlit entrypoint:  streamlit run app/frontend/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script; make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.frontend import nav  # noqa: E402
from app.frontend.theme import inject_css  # noqa: E402
from app.frontend.views import chat, dashboard, documents, evaluation, mcp_info, settings  # noqa: E402

st.set_page_config(page_title="NEXUS AI", page_icon=":material/hub:", layout="wide")
inject_css()

nav.PAGES.update(
    dashboard=st.Page(
        dashboard.render,
        title="Dashboard",
        icon=":material/home:",
        default=True,
    ),
    documents=st.Page(
        documents.render, title="Documents", icon=":material/folder:", url_path="documents"
    ),
    chat=st.Page(chat.render, title="Chat", icon=":material/chat:", url_path="chat"),
    evaluation=st.Page(
        evaluation.render, title="Evaluation", icon=":material/analytics:", url_path="evaluation"
    ),
    mcp=st.Page(mcp_info.render, title="MCP Server", icon=":material/hub:", url_path="mcp"),
    settings=st.Page(
        settings.render, title="Settings", icon=":material/settings:", url_path="settings"
    ),
)
with st.sidebar:
    st.markdown("### NEXUS AI")
    st.caption("Knowledge & Agent Assistant")
st.navigation(list(nav.PAGES.values())).run()
