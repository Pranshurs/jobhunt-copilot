"""Tool contracts: every call is validated, and a bad call becomes an error result."""

from __future__ import annotations

import pytest

from src.tools import TOOL_ARGS, TOOL_SCHEMAS, Toolbox, extract_keywords, skill_keys


@pytest.fixture()
def box():
    return Toolbox("# Jane Doe\nSkills: SQL, Python, pandas", [], output_dir=None)


def _err(result):
    return result.get("error", {}).get("type")


def test_unknown_tool_is_an_error_not_an_exception(box):
    assert _err(box.dispatch("delete_everything", {})) == "unknown_tool"


def test_non_json_arguments_are_rejected_with_a_message_the_model_can_act_on(box):
    result = box.dispatch("extract_keywords", None)
    assert _err(result) == "invalid_arguments"
    assert result["error"]["message"] == "arguments were not a valid JSON object"


def test_tools_receive_validated_values_not_raw_arguments(box):
    # Models often send numbers as strings; the contract coerces "3" -> 3 before the tool runs.
    box.roles_db = [{"id": f"r{i}", "tags": ["python"]} for i in range(5)]
    assert box.dispatch("search_roles", {"keywords": ["python"], "top_k": "3"})["count"] == 3


@pytest.mark.parametrize("name,args", [
    ("extract_keywords", {}),                                   # missing required field
    ("extract_keywords", {"text": ""}),                         # empty
    ("search_roles", {"keywords": ["python"], "top_k": 500}),   # out of range
    ("search_roles", {"keywords": "python"}),                   # wrong type
    ("read_resume", {"path": "/etc/passwd"}),                   # unexpected field
    ("submit_application", {"tailored_resume_markdown": "x"}),  # missing cover letter
])
def test_invalid_arguments_are_rejected(box, name, args):
    assert _err(box.dispatch(name, args)) == "invalid_arguments"


def test_submit_is_accepted_once(box):
    args = {"tailored_resume_markdown": "r", "cover_letter_markdown": "c"}
    assert box.dispatch("submit_application", args)["status"] == "submitted"
    assert _err(box.dispatch("submit_application", args)) == "already_submitted"


def test_a_tool_that_raises_returns_tool_failed(box, monkeypatch):
    monkeypatch.setattr(box, "read_resume", lambda: 1 / 0)
    assert _err(box.dispatch("read_resume", {})) == "tool_failed"


def test_in_memory_toolbox_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    box = Toolbox("resume", [], output_dir=None)
    res = box.dispatch("submit_application", {"tailored_resume_markdown": "r", "cover_letter_markdown": "c"})
    assert res["saved_to"] is None
    assert list(tmp_path.iterdir()) == []


def test_schemas_are_generated_from_the_contracts():
    by_name = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
    assert set(by_name) == set(TOOL_ARGS)
    assert by_name["submit_application"]["parameters"]["required"] == [
        "tailored_resume_markdown", "cover_letter_markdown"]
    assert by_name["search_roles"]["parameters"]["properties"]["top_k"]["maximum"] == 20
    assert by_name["read_resume"]["parameters"]["additionalProperties"] is False


def test_extract_keywords_splits_supported_and_unsupported(box):
    out = box.dispatch("extract_keywords", {"text": "Need SQL, Python, Kubernetes and PyTorch."})
    assert set(out["matched_in_resume"]) == {"Python", "SQL"}
    assert set(out["not_in_resume"]) == {"Kubernetes", "PyTorch"}


def test_ambiguous_words_count_only_in_skill_casing():
    assert skill_keys("go the extra mile, rest assured, node of a graph") == set()
    assert skill_keys("Go, REST and Node services") == {"go", "rest", "node"}
    assert "Go" not in extract_keywords("You'll go the extra mile.")
    assert "Go" in extract_keywords("Services written in Go.")


def test_skill_spellings_normalise():
    assert skill_keys("LLMs and PostgreSQL") == skill_keys("LLM and Postgres") == {"llm", "postgres"}
