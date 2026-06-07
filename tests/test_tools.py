"""Tests for the deterministic tools."""

from __future__ import annotations

from src.tools import Toolbox, extract_keywords, load_roles


def test_extract_keywords_finds_known_skills():
    kws = [k.lower() for k in extract_keywords(
        "We need Python, RAG, LLMs, FastAPI, Docker and strong evaluation."
    )]
    for expected in ["python", "rag", "llm", "fastapi", "docker", "evaluation"]:
        assert any(expected in k for k in kws), f"missing {expected} in {kws}"


def test_search_roles_ranks_by_overlap(tmp_path):
    roles = load_roles("data/roles.json")
    box = Toolbox("resume", roles, str(tmp_path))
    out = box.search_roles(keywords=["rag", "llm", "python"], top_k=3)
    assert out["count"] >= 1
    scores = [r["score"] for r in out["roles"]]
    assert scores == sorted(scores, reverse=True)  # ranked, highest first


def test_search_roles_fallback_when_no_overlap(tmp_path):
    box = Toolbox("resume", load_roles("data/roles.json"), str(tmp_path))
    out = box.search_roles(keywords=["underwater-basket-weaving"], top_k=2)
    assert out["count"] >= 1  # graceful fallback, never empty


def test_submit_application_persists_file(tmp_path):
    box = Toolbox("resume", [], str(tmp_path))
    res = box.submit_application("# tailored", "dear team", ["role-001"])
    assert res["status"] == "submitted"
    assert res["saved_to"].endswith(".md")
    assert box.last_submission["tailored_resume"] == "# tailored"
