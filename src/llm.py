"""LLM clients.

Two implementations behind one interface:

* ``LiveLLM``  — talks to any OpenAI-compatible endpoint (Groq, OpenAI, OpenRouter,
  Together, local Ollama). The ``openai`` SDK is imported lazily so the rest of the
  project (and the eval harness) has zero third-party dependencies in mock mode.
* ``MockLLM``  — a deterministic, offline, *scripted* stand-in, not a model. It walks
  through every tool and submits text assembled only from the candidate's resume, so
  the pipeline, the tools and the eval harness run with no key. Its eval scores measure
  the harness, not model quality.
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
    arguments: Optional[dict[str, Any]]  # None when the model sent arguments that aren't a JSON object


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

        # The SDK retries connection errors, 408/409/429 and 5xx with backoff; the timeout
        # bounds each attempt. A request that still fails raises to the agent, which stops
        # with status "llm_error" instead of hanging or crashing the caller.
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "x",
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )
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
                args = None  # rejected by the toolbox, so the model is told and can retry
            if not isinstance(args, dict):
                args = None
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
            matched = (self._tool_result(messages, "extract_keywords") or {}).get("matched_in_resume", [])
            roles = (self._tool_result(messages, "search_roles") or {}).get("roles", [])
            jd = self._user_jd(messages)
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        cid,
                        "submit_application",
                        {
                            "tailored_resume_markdown": self._compose_resume(resume, matched),
                            "cover_letter_markdown": self._compose_cover_letter(resume, matched, jd),
                            "selected_role_ids": [r.get("id") for r in roles][:4],
                        },
                    )
                ]
            )

        return LLMResponse(content="Application prepared and submitted.")

    # -- deterministic text composition (mock only) -------------------------- #
    # Everything below is assembled from the candidate's resume and the skills the
    # extract_keywords tool found in *both* the job description and the resume. The mock
    # never writes a skill, employer or achievement the resume doesn't contain.
    @staticmethod
    def _company_from_jd(jd: str) -> str:
        m = re.search(r"(?:\bat|@)\s+([A-Z][A-Za-z0-9&\-]*(?: [A-Z][A-Za-z0-9&\-]*){0,3})", jd)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _role_from_jd(jd: str) -> str:
        first = (jd.strip().splitlines() or [""])[0]
        role = re.split(r"\s+(?:at|@)\s+|[.:;(\-–—|!?]", first, maxsplit=1)[0].strip()
        looks_like_title = re.search(
            r"(?i)\b(engineer|developer|scientist|analyst|manager|designer|lead|architect|"
            r"intern|specialist|consultant|researcher)\b", role)
        return f"the {role[:80]} role" if looks_like_title else "this role"

    @staticmethod
    def _name_from_resume(resume: str) -> str:
        m = re.search(r"^#\s*([A-Z][A-Za-z .'\-]+)$", resume, re.MULTILINE)
        return m.group(1).strip() if m else "The candidate"

    def _compose_resume(self, resume: str, matched: list[str]) -> str:
        focus = "\n".join(f"- {k}" for k in matched[:10])
        relevant = (
            f"## Most relevant to this role\n{focus}\n\n" if matched
            else "## Most relevant to this role\n(No listed requirement matched the resume's skills.)\n\n"
        )
        return f"# Tailored Resume\n\n{relevant}## Full resume\n\n{resume.strip()}\n"

    def _compose_cover_letter(self, resume: str, matched: list[str], jd: str) -> str:
        company = self._company_from_jd(jd)
        name = self._name_from_resume(resume)
        title = self._role_from_jd(jd)
        skills = (
            f"The requirements my resume shows experience with are {', '.join(matched[:6])}."
            if matched else "My resume, attached, sets out my experience."
        )
        greeting = f"Dear Hiring Team at {company}," if company else "Dear Hiring Team,"
        return (
            f"{greeting}\n\n"
            f"I'm applying for {title}.\n\n"
            f"{skills} The attached resume has the details and dates.\n\n"
            "I'd welcome the chance to discuss the role.\n\n"
            f"Best regards,\n{name}\n"
        )


def build_llm(settings: Settings) -> BaseLLM:
    return LiveLLM(settings) if settings.mode == "live" else MockLLM(settings)
