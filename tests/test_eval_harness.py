"""The harness itself: grounding metric, expected end states, and the self-check."""

from __future__ import annotations

from eval import run_eval
from eval.metrics import score_case, unsupported_skills


def test_unsupported_skills_are_those_missing_from_the_resume():
    assert unsupported_skills(["Kubernetes and SQL"], "SQL, Python") == ["kubernetes"]
    assert unsupported_skills(["SQL"], "SQL") == []


def test_grounding_fails_on_an_unsupported_skill():
    result = {"status": "submitted", "tailored_resume": "SQL, Kubernetes", "cover_letter": "hi",
              "selected_roles": [{"id": "r"}]}
    checks = score_case(result, {"required_keywords": ["SQL"]}, resume="SQL")
    assert checks["grounding_pass"] is False and checks["unsupported_skills"] == ["kubernetes"]
    assert checks["task_success"] is False


def test_a_run_that_did_not_submit_cannot_pass():
    result = {"status": "max_steps_exceeded", "tailored_resume": "SQL", "cover_letter": "hi",
              "selected_roles": [{"id": "r"}]}
    assert score_case(result, {"required_keywords": ["SQL"]}, resume="SQL")["task_success"] is False


def test_expected_failure_state_passes_only_when_it_happens():
    exp = {"expected_status": "invalid_input"}
    assert score_case({"status": "invalid_input"}, exp)["task_success"] is True
    assert score_case({"status": "submitted"}, exp)["task_success"] is False


def test_mock_passes_every_case_and_the_self_check_discriminates(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODE", "")
    assert run_eval.main(["--policy", "mock", "--require-all-pass", "--out-dir", str(tmp_path)]) == 0
    assert run_eval.main(["--self-check", "--out-dir", str(tmp_path)]) == 0


def test_require_all_pass_fails_for_a_bad_policy(tmp_path):
    assert run_eval.main(["--policy", "fabricating", "--require-all-pass", "--out-dir", str(tmp_path)]) == 1
