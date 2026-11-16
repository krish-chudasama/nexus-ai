"""Deterministic evaluation metrics (pure functions, no LLM, no I/O)."""

from __future__ import annotations

import re
from collections import Counter

_PUNCT = re.compile(r"[^\w\s#:.]")


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 40,000 -> 40000
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip()


def retrieval_metrics(retrieved_files: list[str], expected: list[str]) -> dict[str, float]:
    """``retrieved_files`` is the source filename of each retrieved chunk, in rank order."""
    exp = set(expected)
    first = next((i for i, f in enumerate(retrieved_files, start=1) if f in exp), None)
    found = {f for f in retrieved_files if f in exp}
    return {
        "hit": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "recall": len(found) / len(exp) if exp else 0.0,
        "precision": (
            sum(f in exp for f in retrieved_files) / len(retrieved_files)
            if retrieved_files
            else 0.0
        ),
    }


def keyword_recall(text: str, keywords: list[str]) -> float | None:
    """Fraction of expected keywords present in ``text``. None when there are no keywords."""
    if not keywords:
        return None
    hay = _norm(text)
    return sum(_norm(k) in hay for k in keywords) / len(keywords)


def token_f1(prediction: str, reference: str) -> float:
    p, r = _norm(prediction).split(), _norm(reference).split()
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def citation_metrics(cited_files: list[str], expected: list[str]) -> dict[str, float | None]:
    """precision: share of cited files that are expected. recall: share of expected files cited.
    precision is None when nothing was cited (undefined, not 0)."""
    cited, exp = set(cited_files), set(expected)
    return {
        "precision": len(cited & exp) / len(cited) if cited else None,
        "recall": len(cited & exp) / len(exp) if exp else None,
    }


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
