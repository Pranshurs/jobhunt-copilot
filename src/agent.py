"""The JobHunt Copilot agent: a tool-calling loop over an LLM."""

from __future__ import annotations

import json

from .llm import BaseLLM, LLMResponse
from .prompts import SYSTEM_PROMPT
from .schemas import ApplicationResult, Role, ToolCallTrace
from .tools import TOOL_SCHEMAS, Toolbox

TERMINAL_TOOL = "submit_application"


class JobHuntAgent:
    """Drives the LLM through a read -> analyse -> search -> submit loop."""

    def __init__(self, llm: BaseLLM, toolbox: Toolbox, max_steps: int = 8):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps

    def run(self, job_description: str) -> ApplicationResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": job_description},
        ]

        trace: list[ToolCallTrace] = []
        keywords: list[str] = []
        roles_result: list[dict] = []
        steps_used = 0
        submitted = False

        for step in range(1, self.max_steps + 1):
            steps_used = step
            response: LLMResponse = self.llm.chat(messages, tools=TOOL_SCHEMAS)

            if not response.tool_calls:
                break  # model produced a final message (or gave up) without tools

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                result = self.toolbox.dispatch(tc.name, tc.arguments)
                trace.append(
                    ToolCallTrace(
                        step=step, name=tc.name, arguments=tc.arguments,
                        summary=_summarize(tc.name, result),
                    )
                )
                if tc.name == "extract_keywords":
                    keywords = result.get("keywords", []) or keywords
                elif tc.name == "search_roles":
                    roles_result = result.get("roles", []) or roles_result

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(result),
                    }
                )
                if tc.name == TERMINAL_TOOL:
                    submitted = True

            if submitted:
                break

        return self._assemble(job_description, keywords, roles_result, trace, steps_used)

    def _assemble(self, jd, keywords, roles_result, trace, steps_used) -> ApplicationResult:
        sub = self.toolbox.last_submission or {}
        selected_ids = set(sub.get("selected_role_ids") or [])
        selected_roles = [
            Role(
                id=r.get("id", ""), title=r.get("title", ""), company=r.get("company", ""),
                location=r.get("location", ""), tags=list(r.get("tags", [])),
                url=r.get("url", ""), snippet=r.get("snippet", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in roles_result
            if not selected_ids or r.get("id") in selected_ids
        ]
        return ApplicationResult(
            job_title=_guess_title(jd),
            tailored_resume=sub.get("tailored_resume", ""),
            cover_letter=sub.get("cover_letter", ""),
            selected_roles=selected_roles,
            keywords=keywords,
            trace=trace,
            steps_used=steps_used,
        )


def _summarize(name: str, result: dict) -> str:
    if name == "read_resume":
        return f"{result.get('chars', 0)} chars"
    if name == "extract_keywords":
        return f"{result.get('count', 0)} keywords"
    if name == "search_roles":
        return f"{result.get('count', 0)} roles"
    if name == "submit_application":
        return result.get("status", "")
    return ""


def _guess_title(jd: str) -> str:
    lines = [ln.strip() for ln in jd.strip().splitlines() if ln.strip()]
    return (lines[0] if lines else "Role")[:80]
