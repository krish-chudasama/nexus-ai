"""Shared styling and small UI helpers (kept free of business logic)."""

from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
:root { --nx-accent:#4f46e5; --nx-border:#e5e7eb; --nx-muted:#6b7280; --nx-soft:#f6f7fb; }
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
  Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 1.9rem; margin-bottom: .2rem; }
h2, h3 { font-weight: 650; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: var(--nx-soft); border-right: 1px solid var(--nx-border); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--nx-border); border-radius:12px;
  padding:14px 16px; }
[data-testid="stMetricLabel"] p { color: var(--nx-muted); font-size:.78rem; text-transform:uppercase;
  letter-spacing:.04em; }
[data-testid="stMetricValue"] { font-size:1.5rem !important; line-height:1.25 !important; }
[data-testid="stMetricValue"] *, [data-testid="stMetricLabel"] * { white-space:normal !important;
  overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere; }
.nx-sub { color: var(--nx-muted); font-size: 1.02rem; margin-bottom: 1.4rem; }
.nx-card { border:1px solid var(--nx-border); border-radius:12px; padding:16px 18px; background:#fff;
  height:100%; }
.nx-card h4 { margin:0 0 6px 0; font-size:1rem; }
.nx-card p { margin:0; color:var(--nx-muted); font-size:.9rem; line-height:1.45; }
.nx-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.nx-ok { background:#e8f7ee; color:#146c3a; } .nx-warn { background:#fff4dc; color:#8a5a00; }
.nx-bad { background:#fdeaea; color:#a02323; } .nx-info { background:#eef0ff; color:#3730a3; }
.nx-src { border-left:3px solid var(--nx-accent); background:var(--nx-soft); padding:8px 12px;
  border-radius:0 8px 8px 0; margin:6px 0; font-size:.88rem; }
.nx-src small { color: var(--nx-muted); }
.nx-step { border:1px solid var(--nx-border); border-radius:10px; padding:8px 12px; margin:6px 0;
  font-size:.88rem; background:#fff; }
code { font-size:.85em; }
#MainMenu, footer { visibility:hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="nx-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="nx-pill nx-{kind}">{html.escape(text)}</span>'


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="nx-card"><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def fmt_metric(value: float | int | None, pct: bool = False) -> str:
    """Never show a made-up number: unevaluated metrics render as 'n/a'."""
    if value is None:
        return "n/a"
    return (
        f"{value * 100:.0f}%"
        if pct
        else (f"{value:.2f}" if isinstance(value, float) else str(value))
    )
