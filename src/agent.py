"""The JobHunt Copilot agent: a tool-calling loop over an LLM.

The loop ends in exactly one of these states, recorded as ``ApplicationResult.status``:

* ``submitted``                  — submit_application succeeded (the only success state)
* ``max_steps_exceeded``         — the step budget ran out before a submission
* ``stopped_without_submitting`` — the model replied without calling a tool
* ``llm_error``                  — the LLM request failed after the client's retries
* ``invalid_input``              — the job description was empty or too long; no LLM call made

Tool errors (unknown tool, invalid or non-JSON arguments, a second submit, a tool that
raises) don't end the loop. They are returned to the model as the tool result, so it can
correct itself, and are marked ``ok=False`` in the trace.
"""

from __future__ import annotations

import json

from .llm import BaseLLM, LLMResponse
from .prompts import SYSTEM_PROMPT
from .schemas import ApplicationResult, Role, ToolCallTrace
from .tools import TOOL_SCHEMAS, Toolbox

TERMINAL_TOOL = "submit_application"
MAX_JD_CHARS = 20_000


class JobHuntAgent:
    """Drives the LLM through a read -> analyse -> search -> submit loop."""

    def __init__(self, llm: BaseLLM, toolbox: Toolbox, max_steps: int = 8):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps

    def run(self, job_description: str) -> ApplicationResult:
        jd = (job_description or "").strip()
        if not jd or len(jd) > MAX_JD_CHARS:
            reason = "empty job description" if not jd else f"job description over {MAX_JD_CHARS} characters"
            return self._assemble(jd, [], [], [], 0, "invalid_input", reason)

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": jd},
        ]

        trace: list[ToolCallTrace] = []
        keywords: list[str] = []
        roles_result: list[dict] = []
        steps_used = 0
        status = "max_steps_exceeded"
        error = ""

        for step in range(1, self.max_steps + 1):
            steps_used = step
            try:
                response: LLMResponse = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            except Exception as exc:  # network, auth, rate limit after retries, timeout
                status, error = "llm_error", f"{type(exc).__name__}: {exc}"
                break

            if not response.tool_calls:
                status = "stopped_without_submitting"
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments or {})},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                result = self.toolbox.dispatch(tc.name, tc.arguments)
                ok = "error" not in result
                trace.append(
                    ToolCallTrace(
                        step=step, name=tc.name, arguments=tc.arguments or {},
                        summary=_summarize(tc.name, result), ok=ok,
                    )
                )
                if ok and tc.name == "extract_keywords":
                    keywords = result.get("keywords", []) or keywords
                elif ok and tc.name == "search_roles":
                    roles_result = result.get("roles", []) or roles_result

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(result),
                    }
                )
                if ok and tc.name == TERMINAL_TOOL:
                    status = "submitted"

            if status == "submitted":
                break

        return self._assemble(jd, keywords, roles_result, trace, steps_used, status, error)

    def _assemble(self, jd, keywords, roles_result, trace, steps_used, status, error) -> ApplicationResult:
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
        ] if status == "submitted" else []
        return ApplicationResult(
            job_title=_guess_title(jd),
            tailored_resume=sub.get("tailored_resume", ""),
            cover_letter=sub.get("cover_letter", ""),
            selected_roles=selected_roles,
            keywords=keywords,
            trace=trace,
            steps_used=steps_used,
            status=status,
            error=error,
        )


def _summarize(name: str, result: dict) -> str:
    if "error" in result:
        err = result["error"]
        return f"error {err.get('type')}: {err.get('message', '')}"[:200]
    if name == "read_resume":
        return f"{result.get('chars', 0)} chars"
    if name == "extract_keywords":
        return (f"{result.get('count', 0)} keywords, "
                f"{len(result.get('matched_in_resume', []))} supported by the resume")
    if name == "search_roles":
        suffix = " (unranked fallback)" if result.get("unranked_fallback") else ""
        return f"{result.get('count', 0)} roles{suffix}"
    if name == "submit_application":
        return result.get("status", "")
    return ""


def _guess_title(jd: str) -> str:
    lines = [ln.strip() for ln in jd.strip().splitlines() if ln.strip()]
    return (lines[0] if lines else "Role")[:80]
