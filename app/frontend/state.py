"""Shared Streamlit state helpers."""

from __future__ import annotations

import streamlit as st

from app.frontend.api_client import ApiError, NexusClient


@st.cache_resource
def get_client() -> NexusClient:
    return NexusClient()


def safe_health() -> dict | None:
    """Returns the health payload, or None (after showing a friendly error) if unreachable."""
    try:
        return get_client().health()
    except ApiError as exc:
        st.error(str(exc))
        return None


def show_error(exc: ApiError) -> None:
    if exc.code == "configuration_error":
        st.warning(f"{exc}", icon=":material/key:")
    elif exc.code == "unreachable":
        st.error(str(exc), icon=":material/cloud_off:")
    else:
        st.error(str(exc), icon=":material/error:")
