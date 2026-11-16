"""JSONL evaluation dataset loading with line-level validation errors."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import ValidationFailed


class EvalCase(BaseModel):
    id: str
    question: str = Field(min_length=1)
    expected_answer: str = ""
    expected_sources: list[str] = Field(default_factory=list)  # filenames
    expected_keywords: list[str] = Field(default_factory=list)  # must appear in a correct answer
    answerable: bool = True


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise ValidationFailed(f"Dataset not found: {path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = EvalCase.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise ValidationFailed(
                f"{path.name}:{lineno}: invalid case ({str(exc)[:150]})"
            ) from exc
        if case.id in seen:
            raise ValidationFailed(f"{path.name}:{lineno}: duplicate id '{case.id}'")
        if case.answerable and not case.expected_sources:
            raise ValidationFailed(f"{path.name}:{lineno}: answerable case needs expected_sources")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValidationFailed(f"{path.name} contains no cases.")
    return cases
