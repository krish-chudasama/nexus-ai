"""Optional LLM-as-judge. Never used for the deterministic metrics; opt-in only."""

from __future__ import annotations

import json
import re

from app.core.logging import get_logger
from app.llm.base import LLMClient

log = get_logger(__name__)

JUDGE_PROMPT = """You are a strict grader. Compare the ANSWER to the REFERENCE for the QUESTION.
Score 1-5: 5 = fully correct and complete; 4 = correct, minor omission; 3 = partly correct;
2 = mostly wrong; 1 = wrong or unrelated. Judge factual agreement with the reference only,
not style. Respond with JSON only: {"score": <1-5>, "reason": "<one short sentence>"}"""


def judge_answer(llm: LLMClient, question: str, reference: str, answer: str) -> dict | None:
    """Returns {"score": int, "reason": str} or None if the judge failed / was unparseable."""
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}",
                },
            ],
            temperature=0.0,
            max_tokens=120,
        )
        match = re.search(r"\{.*\}", resp.content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        score = int(data["score"])
        if not 1 <= score <= 5:
            return None
        return {"score": score, "reason": str(data.get("reason", ""))[:200]}
    except Exception:
        log.warning("judge_failed", exc_info=True)
        return None
