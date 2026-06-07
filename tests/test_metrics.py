"""Tests for the evaluation metrics."""

from __future__ import annotations

from eval.metrics import (
    grounding_ok,
    keyword_coverage,
    score_case,
    structure_complete,
    wilson_interval,
)


def test_keyword_coverage():
    assert keyword_coverage("python and rag", ["python", "rag"]) == 1.0
    assert keyword_coverage("python only", ["python", "rag"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0


def test_grounding_ok():
    assert grounding_ok(["honest resume"], ["PhD", "Google"]) is True
    assert grounding_ok(["I have a PhD"], ["PhD"]) is False


def test_structure_complete():
    good = {"tailored_resume": "x", "cover_letter": "y", "selected_roles": [{"id": "r1"}]}
    assert structure_complete(good, ["tailored_resume", "cover_letter", "role_matches"], 1) is True
    bad = {"tailored_resume": "", "cover_letter": "y", "selected_roles": []}
    assert structure_complete(bad, ["tailored_resume", "cover_letter", "role_matches"], 1) is False


def test_score_case_passes_for_good_output():
    result = {
        "tailored_resume": "Python, RAG, LLM, FastAPI, evaluation, embeddings",
        "cover_letter": "I build and evaluate AI.",
        "selected_roles": [{"id": "role-001"}],
    }
    expectations = {
        "required_keywords": ["python", "rag", "llm", "fastapi", "evaluation"],
        "forbidden_claims": ["PhD"],
        "min_role_matches": 1,
    }
    checks = score_case(result, expectations)
    assert checks["task_success"] is True


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(3, 3)
    assert 0.0 <= lo <= hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
