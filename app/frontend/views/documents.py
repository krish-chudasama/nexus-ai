from __future__ import annotations

import pandas as pd
import streamlit as st

from app.frontend.api_client import ApiError
from app.frontend.state import get_client, safe_health, show_error
from app.frontend.theme import fmt_bytes, page_header


@st.dialog("Delete document?")
def _confirm_delete(doc: dict) -> None:
    st.write(f"**{doc['filename']}** and its {doc['num_chunks']} indexed chunks will be removed.")
    left, right = st.columns(2)
    if left.button("Delete", type="primary", width="stretch"):
        try:
            get_client().delete_document(doc["id"])
            st.rerun()
        except ApiError as exc:
            show_error(exc)
    if right.button("Cancel", width="stretch"):
        st.rerun()


def _upload_panel() -> None:
    st.session_state.setdefault("upload_version", 0)
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_version}",
        help="PDF, DOCX, TXT or Markdown. Scanned (image-only) PDFs are not supported.",
    )
    if files and st.button(f"Index {len(files)} file(s)", type="primary"):
        results = []
        with st.status("Indexing documents…", expanded=True) as status:
            for f in files:
                st.write(f"Processing **{f.name}**…")
                try:
                    doc = get_client().upload(f.name, f.getvalue())["document"]
                    results.append(
                        ("ok", f"{doc['filename']} — {doc['num_chunks']} chunks indexed")
                    )
                except ApiError as exc:
                    results.append(("err", f"{f.name} — {exc}"))
            failed = sum(kind == "err" for kind, _ in results)
            status.update(
                label="Done" if not failed else f"Done with {failed} problem(s)",
                state="complete" if not failed else "error",
                expanded=False,
            )
        st.session_state.upload_results = results
        st.session_state.upload_version += 1  # clears the uploader
        st.rerun()

    for kind, msg in st.session_state.pop("upload_results", []):
        (st.success if kind == "ok" else st.error)(msg)


def render() -> None:
    page_header("Documents", "Upload files and manage your knowledge base.")
    if safe_health() is None:
        st.stop()
    _upload_panel()

    st.subheader("Library")
    try:
        data = get_client().documents()
    except ApiError as exc:
        show_error(exc)
        st.stop()
    docs = data["documents"]
    if not docs:
        st.info(
            "No documents yet. Upload a file above to get started.", icon=":material/upload_file:"
        )
        return

    query = st.text_input("Filter", placeholder="Filter by filename…", label_visibility="collapsed")
    shown = [d for d in docs if query.lower() in d["filename"].lower()]
    st.caption(f"{len(shown)} of {len(docs)} documents · {data['total_chunks']} chunks")

    frame = pd.DataFrame(
        [
            {
                "File": d["filename"],
                "Type": d["file_type"].upper(),
                "Pages": d["num_pages"],
                "Chunks": d["num_chunks"],
                "Size": fmt_bytes(d["size_bytes"]),
                "Uploaded": d["uploaded_at"].replace("T", " ")[:16],
                "ID": d["id"],
            }
            for d in shown
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")

    st.markdown("##### Details")
    for d in shown:
        with st.expander(d["filename"]):
            c1, c2 = st.columns([4, 1])
            c1.caption(
                f"id `{d['id']}` · {d['num_chars']:,} characters · sha256 `{d['sha256'][:12]}…`"
            )
            if c2.button("Delete", key=f"del_{d['id']}", icon=":material/delete:", width="stretch"):
                _confirm_delete(d)
            st.text(d["preview"] + ("…" if d["num_chars"] > len(d["preview"]) else ""))
