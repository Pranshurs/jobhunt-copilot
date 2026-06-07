"""LLM clients.

Two implementations behind one interface:

* ``LiveLLM``  — talks to any OpenAI-compatible endpoint (Groq, OpenAI, OpenRouter,
  Together, local Ollama). The ``openai`` SDK is imported lazily so the rest of the
  project (and the eval harness) has zero third-party dependencies in mock mode.
* ``MockLLM``  — a deterministic, offline stand-in. It simulates an agent that walks
  through every tool and then submits, so the whole pipeline + eval run with no key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLM:
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Live client                                                                 #
# --------------------------------------------------------------------------- #
class LiveLLM(BaseLLM):
    def __init__(self, settings: Settings):
        from openai import OpenAI  # lazy import: only needed in live mode

        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key or "x")
        self.model = settings.model
        self.temperature = settings.temperature

    def chat(self, messages, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return LLMResponse(content=msg.content, tool_calls=calls)


# --------------------------------------------------------------------------- #
# Offline mock client                                                         #
# --------------------------------------------------------------------------- #
class MockLLM(BaseLLM):
    """Deterministic agent simulation for offline runs, demos, CI, and evaluation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._counter = 0

    # -- helpers to read prior state out of the message history -------------- #
    @staticmethod
    def _tool_result(messages: list[dict], name: str) -> Optional[dict]:
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("name") == name:
                try:
                    return json.loads(m.get("content") or "{}")
                except json.JSONDecodeError:
                    return {}
        return None

    def _called(self, messages: list[dict], name: str) -> bool:
        return self._tool_result(messages, name) is not None

    @staticmethod
    def _user_jd(messages: list[dict]) -> str:
        for m in messages:
            if m.get("role") == "user":
                return m.get("content") or ""
        return ""

    # -- the scripted policy -------------------------------------------------- #
    def chat(self, messages, tools=None) -> LLMResponse:
        self._counter += 1
        cid = f"call_{self._counter}"

        if not self._called(messages, "read_resume"):
            return LLMResponse(tool_calls=[ToolCall(cid, "read_resume", {})])

        if not self._called(messages, "extract_keywords"):
            jd = self._user_jd(messages)
            return LLMResponse(tool_calls=[ToolCall(cid, "extract_keywords", {"text": jd})])

        if not self._called(messages, "search_roles"):
            kw = self._tool_result(messages, "extract_keywords") or {}
            return LLMResponse(
                tool_calls=[ToolCall(cid, "search_roles", {"keywords": kw.get("keywords", []), "top_k": 4})]
            )

        if not self._called(messages, "submit_application"):
            resume = (self._tool_result(messages, "read_resume") or {}).get("resume", "")
            keywords = (self._tool_result(messages, "extract_keywords") or {}).get("keywords", [])
            roles = (self._tool_result(messages, "search_roles") or {}).get("roles", [])
            jd = self._user_jd(messages)
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        cid,
                        "submit_application",
                        {
                            "tailored_resume_markdown": self._compose_resume(resume, keywords),
                            "cover_letter_markdown": self._compose_cover_letter(resume, keywords, jd),
                            "selected_role_ids": [r.get("id") for r in roles][:4],
                        },
                    )
                ]
            )

        return LLMResponse(content="Application prepared and submitted.")

    # -- deterministic text composition (mock only) -------------------------- #
    @staticmethod
    def _company_from_jd(jd: str) -> str:
        m = re.search(r"(?:\bat|@)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", jd)
        return m.group(1).strip().rstrip(".") if m else "your team"

    @staticmethod
    def _name_from_resume(resume: str) -> str:
        m = re.search(r"#\s*([A-Z][A-Za-z .]+)", resume)
        return m.group(1).strip() if m else "The candidate"

    def _compose_resume(self, resume: str, keywords: list[str]) -> str:
        kw_line = ", ".join(keywords[:12]) if keywords else "the role's core skills"
        focus = "\n".join(f"- {k}" for k in keywords[:8]) or "- Core engineering fundamentals"
        return (
            "# Tailored Resume\n\n"
            f"**Profile (tailored to this role):** AI/ML engineer with hands-on experience in "
            f"{kw_line}.\n\n"
            "## Most relevant to this role\n"
            f"{focus}\n\n"
            "## Full resume\n\n"
            f"{resume.strip()}\n"
        )

    def _compose_cover_letter(self, resume: str, keywords: list[str], jd: str) -> str:
        company = self._company_from_jd(jd)
        name = self._name_from_resume(resume)
        kws = ", ".join(keywords[:5]) if keywords else "the skills this role needs"
        return (
            f"Dear Hiring Team at {company},\n\n"
            f"I'm applying for this role because it maps directly to what I build and measure: {kws}. "
            "Unusually for an early-career engineer, I don't just ship AI features — I evaluate them. "
            "I ran LLM reasoning evaluation at Outlier and built statistical rigour into a full ML "
            "research platform, so I can prove a system works, not just demo it.\n\n"
            "My background spans LLMs, RAG, agents, and AI automation across Python, FastAPI, Docker "
            "and n8n. I move fast on shipped, deployed work and care about correctness — grounding, "
            "task-success evaluation, and clear measurement.\n\n"
            "I'd welcome the chance to show how quickly I can contribute.\n\n"
            f"Best regards,\n{name}\n"
        )


def build_llm(settings: Settings) -> BaseLLM:
    return LiveLLM(settings) if settings.mode == "live" else MockLLM(settings)
