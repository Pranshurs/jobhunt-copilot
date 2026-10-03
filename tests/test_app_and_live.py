"""Web API limits and privacy; live client wiring (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app as app_module
from src.config import Settings
from src.llm import LiveLLM


def test_api_labels_mock_output_and_persists_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setattr(app_module, "ROOT", tmp_path, raising=False)
    client = TestClient(app_module.app)
    body = client.post("/api/run", json={"job_description": "Data Analyst at X. SQL, Python.",
                                         "resume": "# Jane Doe\nSQL, Python"}).json()
    assert body["status"] == "submitted"
    assert body["mode"] == "mock" and "not a model" in body["model"]
    assert not (tmp_path / "outputs").exists()


def test_api_rejects_empty_and_oversized_input():
    client = TestClient(app_module.app)
    assert client.post("/api/run", json={"job_description": ""}).status_code == 422
    assert client.post("/api/run", json={"job_description": "x" * 20_001}).status_code == 422
    assert client.post("/api/run", json={"job_description": "ok", "resume": "x" * 50_001}).status_code == 422


def _settings(**kw):
    base = dict(mode="live", base_url="http://localhost:9/v1", api_key="k", model="m",
                temperature=0.0, max_steps=8, timeout_s=12.5, max_retries=3)
    return Settings(**{**base, **kw})


def test_live_client_passes_timeout_and_retries(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    LiveLLM(_settings())
    assert seen["timeout"] == 12.5 and seen["max_retries"] == 3


def test_live_client_marks_non_json_tool_arguments(monkeypatch):
    def tc(i, args):
        return SimpleNamespace(id=i, function=SimpleNamespace(name="extract_keywords", arguments=args))

    message = SimpleNamespace(content=None, tool_calls=[tc("a", "{not json"), tc("b", json.dumps([1])),
                                                        tc("c", json.dumps({"text": "SQL"}))])
    completions = SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=message)]))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    calls = LiveLLM(_settings()).chat([{"role": "user", "content": "x"}]).tool_calls
    assert [c.arguments for c in calls] == [None, None, {"text": "SQL"}]
