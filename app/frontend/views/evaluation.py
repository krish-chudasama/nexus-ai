from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.frontend.api_client import ApiError
from app.frontend.state import get_client, safe_health, show_error
from app.frontend.theme import fmt_metric, page_header

HEADLINE = [
    ("Hit rate", "hit_rate", True),
    ("MRR", "mrr", False),
    ("Recall", "recall", True),
    ("Abstention acc.", "abstention_accuracy", True),
    ("Answer acc.", "answer_accuracy", True),
    ("Citation prec.", "citation_precision", True),
]


def _metrics_row(summary: dict) -> None:
    cols = st.columns(len(HEADLINE))
    for col, (label, key, pct) in zip(cols, HEADLINE, strict=True):
        col.metric(label, fmt_metric(summary.get(key), pct))


def _poll(run_id: str) -> dict | None:
    with st.status("Running evaluation…", expanded=True) as status:
        for _ in range(240):
            body = get_client().eval_status(run_id)
            if body.get("status") != "running":
                if body.get("status") == "failed":
                    status.update(label="Evaluation failed", state="error")
                    st.error(body.get("error") or "Evaluation failed.")
                    return None
                status.update(label="Evaluation complete", state="complete", expanded=False)
                return body
            time.sleep(1)
        status.update(label="Still running — check back shortly", state="error")
    return None


def _report(report: dict) -> None:
    cfg = report["config"]
    st.caption(
        f"Run `{report['run_id']}` · {report['mode']} · dataset `{report['dataset']}` · "
        f"embedding `{cfg['embedding']}` · top-k {cfg['top_k']} · "
        f"{cfg['documents_indexed']} docs · {report['duration_s']}s"
    )
    _metrics_row(report["summary"])
    if report["mode"] == "retrieval-only":
        st.caption(
            "Retrieval-only run: answer and citation metrics are shown as n/a because "
            "no LLM was called."
        )

    failures = report["failures"]
    st.subheader(f"Failures ({len(failures)})")
    if failures:
        st.dataframe(pd.DataFrame(failures), hide_index=True, width="stretch")
    else:
        st.success("No failing cases in this run.")

    st.subheader("Per-case results")
    rows = []
    for c in report["cases"]:
        r = c.get("retrieval") or {}
        rows.append(
            {
                "id": c["id"],
                "question": c["question"],
                "answerable": c["answerable"],
                "top source": (c.get("retrieved") or [{}])[0].get("file"),
                "hit": r.get("hit"),
                "mrr": r.get("mrr"),
                "correct": (c.get("answer_metrics") or {}).get("correct"),
                "failed": bool(c["failures"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Raw report JSON"):
        st.json(report, expanded=False)


def render() -> None:
    page_header("Evaluation", "Measure retrieval and answer quality against a JSONL dataset.")
    health = safe_health()
    if health is None:
        st.stop()
    client = get_client()

    with st.container(border=True):
        try:
            datasets = client.eval_datasets()
        except ApiError as exc:
            show_error(exc)
            st.stop()
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        dataset = c1.selectbox("Dataset", datasets or ["(none found)"])
        use_llm = c2.toggle("Use LLM", help="Calls the LLM. Needs LLM_API_KEY.")
        use_judge = c3.toggle("LLM judge", disabled=not use_llm, help="Optional; extra LLM calls.")
        top_k = c4.number_input("Top-k", 1, 20, 5)
        st.caption(
            "Evaluations index the bundled sample corpus into an isolated store, so your own "
            "library is never touched."
        )
        if st.button("Run evaluation", type="primary", disabled=not datasets):
            try:
                job = client.eval_run(
                    {
                        "dataset": dataset,
                        "use_llm": use_llm,
                        "use_judge": use_judge and use_llm,
                        "top_k": int(top_k),
                    }
                )
                done = _poll(job["run_id"])
                if done:
                    st.session_state.eval_selected = job["run_id"]
            except ApiError as exc:
                show_error(exc)

    try:
        runs = client.eval_runs()
    except ApiError as exc:
        show_error(exc)
        return
    if not runs:
        st.info("No saved runs yet. Run an evaluation above.", icon=":material/analytics:")
        return

    ids = [r["run_id"] for r in runs]
    selected = st.selectbox(
        "Saved runs",
        ids,
        index=ids.index(st.session_state.get("eval_selected", ids[0]))
        if st.session_state.get("eval_selected") in ids
        else 0,
    )
    if len(runs) > 1:
        trend = pd.DataFrame(
            [
                {"run": r["run_id"], "hit_rate": r["summary"].get("hit_rate"),
                 "mrr": r["summary"].get("mrr")}
                for r in reversed(runs)
            ]
        ).set_index("run")  # fmt: skip
        st.caption("Retrieval metrics across saved runs (oldest → newest)")
        st.line_chart(trend)
    try:
        _report(client.eval_status(selected))
    except ApiError as exc:
        show_error(exc)
