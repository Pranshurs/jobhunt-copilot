"""End-to-end test of the agent loop in offline mock mode."""

from __future__ import annotations

from pathlib import Path

from eval.metrics import score_case
from src.agent import JobHuntAgent
from src.config import Settings
from src.llm import build_llm
from src.tools import Toolbox, load_roles

MOCK = Settings(mode="mock", base_url="", api_key="", model="mock", temperature=0.0, max_steps=8)


def _agent(out_dir):
    resume = Path("data/resume.md").read_text(encoding="utf-8")
    roles = load_roles("data/roles.json")
    return JobHuntAgent(build_llm(MOCK), Toolbox(resume, roles, str(out_dir)), MOCK.max_steps)


def test_agent_produces_complete_application(tmp_path):
    result = _agent(tmp_path).run(
        "AI Engineer at Acme. RAG, LLMs, Python, FastAPI, evaluation, embeddings."
    )
    assert result.tailored_resume.strip()
    assert result.cover_letter.strip()
    assert len(result.selected_roles) >= 1
    assert result.keywords
    assert result.steps_used > 0


def test_agent_uses_all_tools_in_order(tmp_path):
    result = _agent(tmp_path).run("Applied AI Engineer. Agents, evaluation, Python, LLM, RAG.")
    names = [t.name for t in result.trace]
    assert names[0] == "read_resume"
    assert "extract_keywords" in names
    assert "search_roles" in names
    assert names[-1] == "submit_application"


def test_agent_output_passes_eval(tmp_path):
    result = _agent(tmp_path).run(
        "AI Engineer at Acme. RAG, LLMs, Python, FastAPI, evaluation, embeddings."
    ).to_dict()
    checks = score_case(result, {
        "required_keywords": ["python", "rag", "llm", "fastapi", "evaluation"],
        "forbidden_claims": ["PhD", "Google", "Meta"],
        "min_role_matches": 1,
    })
    assert checks["task_success"] is True
