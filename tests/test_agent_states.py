"""The agent ends in an explicit state, and the mock never fabricates."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.metrics import unsupported_skills
from eval.policies import NeverSubmitMock
from src.agent import JobHuntAgent
from src.config import Settings
from src.llm import BaseLLM, LLMResponse, MockLLM, ToolCall
from src.tools import Toolbox, load_roles

SETTINGS = Settings(mode="mock", base_url="", api_key="", model="mock", temperature=0.0, max_steps=8)
JANE = Path("data/sample_resumes/jane_doe_data_analyst.md").read_text(encoding="utf-8")
DEFAULT = Path("data/resume.md").read_text(encoding="utf-8")


def _run(llm, jd, resume=JANE, max_steps=8):
    return JobHuntAgent(llm, Toolbox(resume, load_roles(), output_dir=None), max_steps).run(jd)


def test_mock_claims_only_skills_the_resume_has():
    r = _run(MockLLM(SETTINGS), "Platform Engineer at Initech. Kubernetes, Go, AWS, Terraform, Python, SQL.")
    assert r.status == "submitted"
    assert unsupported_skills([r.tailored_resume, r.cover_letter], JANE) == []
    assert "Python" in r.tailored_resume and "Kubernetes" not in r.tailored_resume


def test_mock_never_copies_the_default_candidate_into_another_resume():
    r = _run(MockLLM(SETTINGS), "AI Engineer at Acme. RAG, LLMs, Python, evaluation.")
    text = r.tailored_resume + r.cover_letter
    for leaked in ("Outlier", "Ylemis", "Pranshu", "Rust", "n8n"):
        assert leaked not in text
    assert "Jane Doe" in r.cover_letter


def test_never_submitting_ends_max_steps_exceeded():
    r = _run(NeverSubmitMock(SETTINGS), "Data Analyst. SQL.", max_steps=5)
    assert (r.status, r.steps_used) == ("max_steps_exceeded", 5)
    assert r.tailored_resume == "" and r.selected_roles == []


class _Scripted(BaseLLM):
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, tools=None):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_llm_failure_ends_llm_error_without_raising():
    r = _run(_Scripted([TimeoutError("read timed out")]), "Data Analyst. SQL.")
    assert r.status == "llm_error" and "TimeoutError" in r.error


def test_plain_reply_ends_stopped_without_submitting():
    r = _run(_Scripted([LLMResponse(content="Here is your letter...")]), "Data Analyst. SQL.")
    assert r.status == "stopped_without_submitting"


def test_bad_tool_call_is_reported_and_the_model_can_recover():
    submit = {"tailored_resume_markdown": "SQL", "cover_letter_markdown": "Dear team"}
    r = _run(_Scripted([
        LLMResponse(tool_calls=[ToolCall("1", "submit_application", None)]),      # non-JSON args
        LLMResponse(tool_calls=[ToolCall("2", "shell", {"cmd": "ls"})]),          # unknown tool
        LLMResponse(tool_calls=[ToolCall("3", "submit_application", submit)]),
    ]), "Data Analyst. SQL.")
    assert r.status == "submitted"
    assert [t.ok for t in r.trace] == [False, False, True]
    assert r.trace[0].summary.startswith("error invalid_arguments")


@pytest.mark.parametrize("jd", ["", "   ", "x" * 20_001], ids=["empty", "blank", "oversized"])
def test_invalid_job_description_never_reaches_the_llm(jd):
    r = _run(_Scripted([]), jd)  # _Scripted([]) would raise if called
    assert r.status == "invalid_input" and r.steps_used == 0
