"""Registry of navigation pages, populated by app.py so views can link to each other."""

from __future__ import annotations

import streamlit as st

PAGES: dict[str, st.Page] = {}
