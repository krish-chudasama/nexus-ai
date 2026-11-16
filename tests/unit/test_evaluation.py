import json

import pytest

from app.core.errors import ConfigurationError, NexusError, ValidationFailed
from app.embeddings.embedders import HashingEmbedder
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import (
    citation_metrics,
    keyword_recall,
    mean,
    retrieval_metrics,
    token_f1,
)
from app.evaluation.runner import EvalRunner, format_summary, save_report
from tests.conftest import CORPUS, FakeLLM

SAMPLE = CORPUS.parent / "sample.jsonl"


class TestMetrics:
    def test_retrieval_metrics(self):
        m = retrieval_metrics(["b.md", "a.md", "a.md"], ["a.md"])
        assert m == {"hit": 1.0, "mrr": 0.5, "recall": 1.0, "precision": pytest.approx(2 / 3)}
        miss = retrieval_metrics(["b.md"], ["a.md"])
        assert miss["hit"] == 0.0 and miss["mrr"] == 0.0 and miss["recall"] == 0.0
        assert retrieval_metrics([], ["a.md"])["precision"] == 0.0

    def test_recall_with_multiple_expected(self):
        assert retrieval_metrics(["a.md"], ["a.md", "b.md"])["recall"] == 0.5

    def test_keyword_recall_normalises_numbers_and_case(self):
        assert keyword_recall("The stipend is 40000 rupees", ["40,000"]) == 1.0
        assert keyword_recall("It is THREE days", ["three", "days"]) == 1.0
        assert keyword_recall("nothing here", ["three", "days"]) == 0.0
        assert keyword_recall("x", []) is None

    def test_token_f1(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0
        assert token_f1("dog", "cat") == 0.0
        assert 0 < token_f1("the cat", "the cat sat down") < 1

    def test_citation_metrics(self):
        assert citation_metrics(["a.md", "b.md"], ["a.md"]) == {"precision": 0.5, "recall": 1.0}
        assert citation_metrics([], ["a.md"]) == {"precision": None, "recall": 0.0}

    def test_mean_ignores_none(self):
        assert mean([1.0, None, 0.0]) == 0.5 and mean([None]) is None


class TestDataset:
    def test_sample_dataset_is_valid(self):
        cases = load_dataset(SAMPLE)
        assert len(cases) == 10 and sum(not c.answerable for c in cases) == 1

    def test_validation_errors(self, tmp_path):
        def write(text):
            p = tmp_path / "d.jsonl"
            p.write_text(text)
            return p

        with pytest.raises(ValidationFailed, match="invalid case"):
            load_dataset(write("not json"))
        with pytest.raises(ValidationFailed, match="duplicate id"):
            row = '{"id":"a","question":"q","expected_sources":["x.md"]}\n'
            load_dataset(write(row * 2))
        with pytest.raises(ValidationFailed, match="needs expected_sources"):
            load_dataset(write('{"id":"a","question":"q"}'))
        with pytest.raises(ValidationFailed, match="no cases"):
            load_dataset(write("# only a comment\n"))
        with pytest.raises(ValidationFailed, match="not found"):
            load_dataset(tmp_path / "missing.jsonl")


@pytest.fixture
def runner(settings):
    return EvalRunner(settings, embedder=HashingEmbedder())


def _one(**extra):
    return {
        "id": "a",
        "question": "Pro plan price per seat",
        "expected_sources": ["product_faq.md"],
        "expected_keywords": ["29"],
        **extra,
    }


def _cases(tmp_path, rows):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


class TestRunner:
    def test_retrieval_only_run(self, runner, settings):
        report = runner.run(SAMPLE, corpus=CORPUS)
        s = report["summary"]
        assert report["mode"] == "retrieval-only" and s["cases"] == 10 and s["answerable"] == 9
        # hashing embedder + the low test threshold: every answerable question finds its source
        assert s["hit_rate"] == 1.0 and 0 < s["mrr"] <= 1.0
        # ...but this threshold cannot reject the off-topic question; that must be REPORTED
        assert any(f["type"] == "should_abstain" and f["case"] == "q10" for f in report["failures"])
        assert s["abstention_accuracy"] == 0.0
        assert "answer_accuracy" not in s  # no LLM -> answer metrics are not fabricated

    def test_isolated_index_leaves_live_library_untouched(self, runner, settings):
        from app.services.container import build_services

        runner.run(SAMPLE, corpus=CORPUS)
        assert build_services(settings, HashingEmbedder()).documents.list().documents == []

    def test_llm_mode_scores_answers_and_citations(self, settings, tmp_path):
        ds = _cases(
            tmp_path,
            [
                {"id": "ok", "question": "How much does the Pro plan cost per seat?",
                 "expected_answer": "29 dollars", "expected_sources": ["product_faq.md"],
                 "expected_keywords": ["29"]},
                {"id": "wrong", "question": "What is the API rate limit on the Pro plan?",
                 "expected_answer": "600 per minute", "expected_sources": ["product_faq.md"],
                 "expected_keywords": ["600"]},
                {"id": "none", "question": "capital of Australia", "answerable": False},
            ],
        )  # fmt: skip
        llm = FakeLLM(
            ["It costs 29 dollars [1].", "Around 100 per minute [1].", "INSUFFICIENT_CONTEXT"]
        )
        report = EvalRunner(settings, HashingEmbedder(), lambda: llm).run(
            ds, use_llm=True, corpus=CORPUS
        )
        s = report["summary"]
        assert s["answer_accuracy"] == 0.5
        assert s["citation_precision"] == 1.0 and s["abstention_accuracy"] == 1.0
        assert [f["type"] for f in report["failures"]] == ["answer_incorrect"]
        assert report["cases"][0]["answer_metrics"]["correct"] is True

    def test_uncited_answer_flagged(self, settings, tmp_path):
        ds = _cases(tmp_path, [_one()])
        llm = FakeLLM(["29 dollars, no citation."])
        report = EvalRunner(settings, HashingEmbedder(), lambda: llm).run(
            ds, use_llm=True, corpus=CORPUS
        )
        assert [f["type"] for f in report["failures"]] == ["no_citations"]

    def test_judge_is_optional_and_parsed(self, settings, tmp_path):
        ds = _cases(tmp_path, [_one(expected_answer="29 dollars")])
        llm = FakeLLM(["29 dollars [1].", 'Sure: {"score": 5, "reason": "matches"}'])
        report = EvalRunner(settings, HashingEmbedder(), lambda: llm).run(
            ds, use_llm=True, use_judge=True, corpus=CORPUS
        )
        assert report["summary"]["judge_mean_score"] == 5.0

    def test_unparseable_judge_is_skipped_not_fabricated(self, settings, tmp_path):
        ds = _cases(tmp_path, [_one()])
        llm = FakeLLM(["29 dollars [1].", "I refuse to answer in JSON"])
        report = EvalRunner(settings, HashingEmbedder(), lambda: llm).run(
            ds, use_llm=True, use_judge=True, corpus=CORPUS
        )
        assert report["summary"]["judge_mean_score"] is None

    def test_llm_mode_without_key_fails_fast(self, runner):
        from app.core.config import Settings

        keyless = Settings(_env_file=None, embedding_backend="hashing", llm_api_key="")
        with pytest.raises(ConfigurationError):
            EvalRunner(keyless, HashingEmbedder()).run(SAMPLE, use_llm=True, corpus=CORPUS)

    def test_judge_requires_llm(self, runner):
        with pytest.raises(NexusError, match="requires"):
            runner.run(SAMPLE, use_judge=True, corpus=CORPUS)

    def test_report_persistence_and_summary_format(self, runner, tmp_path):
        report = runner.run(SAMPLE, corpus=CORPUS)
        path = save_report(report, tmp_path / "res")
        assert json.loads(path.read_text())["run_id"] == report["run_id"]
        text = format_summary(report)
        assert "hit_rate" in text and "Failures" in text
