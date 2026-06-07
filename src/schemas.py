"""Typed data structures passed between the agent, tools, and the eval harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Role:
    id: str
    title: str
    company: str
    location: str
    tags: list[str] = field(default_factory=list)
    url: str = ""
    snippet: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallTrace:
    """A single step in the agent's reasoning loop, for transparency and debugging."""

    step: int
    name: str
    arguments: dict[str, Any]
    summary: str = ""


@dataclass
class ApplicationResult:
    job_title: str
    tailored_resume: str
    cover_letter: str
    selected_roles: list[Role] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    trace: list[ToolCallTrace] = field(default_factory=list)
    steps_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
