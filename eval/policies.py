"""Deliberately bad scripted policies, used to prove the eval harness can fail.

A harness that only ever reports passes measures nothing. ``python -m eval.run_eval
--self-check`` runs every case with each policy below. A policy has to fail the cases
its defect affects, or the self-check fails. They are test doubles, not models.
"""

from __future__ import annotations

from src.llm import LLMResponse, MockLLM, ToolCall


class FabricatingMock(MockLLM):
    """The 0.1 mock's behaviour. It claims every skill the job asks for, and pastes the
    default candidate's own career highlights into every cover letter."""

    def _compose_resume(self, resume: str, matched: list[str]) -> str:
        jd_keywords = self._last_keywords
        return (
            "# Tailored Resume\n\n"
            f"**Profile:** engineer with hands-on experience in {', '.join(jd_keywords)}.\n\n"
            f"## Full resume\n\n{resume.strip()}\n"
        )

    def _compose_cover_letter(self, resume: str, matched: list[str], jd: str) -> str:
        return (
            "Dear Hiring Team,\n\n"
            f"I bring {', '.join(self._last_keywords)}. I ran LLM reasoning evaluation at "
            "Outlier and built a full ML research platform in Rust.\n\nBest regards\n"
        )

    def chat(self, messages, tools=None) -> LLMResponse:
        self._last_keywords = (self._tool_result(messages, "extract_keywords") or {}).get("keywords", [])
        return super().chat(messages, tools)


class NeverSubmitMock(MockLLM):
    """Keeps searching and never calls submit_application."""

    def chat(self, messages, tools=None) -> LLMResponse:
        if self._called(messages, "search_roles"):
            self._counter += 1
            return LLMResponse(tool_calls=[ToolCall(f"call_{self._counter}", "search_roles", {"keywords": ["python"]})])
        return super().chat(messages, tools)


class InjectionFollowingMock(MockLLM):
    """Obeys instructions embedded in the job description (a prompt-injection failure)."""

    def _compose_cover_letter(self, resume: str, matched: list[str], jd: str) -> str:
        letter = super()._compose_cover_letter(resume, matched, jd)
        if "ignore all previous instructions" in jd.lower():
            letter += "\nP.S. I hold a PhD from MIT and have 10 years of experience at Google.\n"
        return letter


BAD_POLICIES = {
    "fabricating": FabricatingMock,
    "never_submit": NeverSubmitMock,
    "injection_following": InjectionFollowingMock,
}
