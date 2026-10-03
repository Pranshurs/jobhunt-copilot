"""Agent tools, their typed argument contracts, and the OpenAI-style schemas.

The tools are deliberately concrete and deterministic (resume I/O, keyword
extraction, role retrieval/ranking, and a terminal "submit" action). The agent
LLM decides *when* to call them and composes the final deliverables; the tools
do the grunt work.

Every call is validated against a pydantic model before it runs. A bad call never
raises into the agent loop: it returns ``{"error": {"type", "message"}}``, which the
model sees as the tool result and can correct. The JSON schemas the model is given
are generated from the same models, so the contract and the validation can't drift.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# --------------------------------------------------------------------------- #
# Keyword extraction                                                          #
# --------------------------------------------------------------------------- #
SKILL_VOCAB = {
    # languages / runtimes
    "python", "rust", "go", "java", "javascript", "typescript", "sql", "bash",
    # web / services
    "fastapi", "flask", "django", "node", "react", "api", "apis", "rest",
    "microservices", "graphql",
    # infra / data
    "docker", "kubernetes", "redis", "postgres", "postgresql", "sqlite",
    "mongodb", "aws", "gcp", "azure", "linux", "git", "ci/cd", "etl",
    "data pipeline",
    # ml / ai
    "llm", "llms", "rag", "agents", "agentic", "embeddings", "vector database",
    "vector db", "chroma", "qdrant", "pinecone", "pgvector", "fine-tuning",
    "prompt engineering", "evaluation", "eval", "hallucination", "nlp",
    "computer vision", "machine learning", "deep learning", "pytorch",
    "tensorflow", "scikit-learn", "pandas", "numpy", "mlops", "openai",
    "anthropic", "hugging face", "transformers", "langchain", "llamaindex",
    "n8n", "automation", "statistics", "backtesting", "slack",
}

_MULTIWORD = sorted(
    (s for s in SKILL_VOCAB if " " in s or "/" in s),
    key=len,
    reverse=True,
)

_CANON = {
    "llm": "LLM", "llms": "LLMs", "rag": "RAG", "nlp": "NLP", "api": "API",
    "apis": "API", "rest": "REST", "sql": "SQL", "aws": "AWS", "gcp": "GCP",
    "azure": "Azure", "ci/cd": "CI/CD", "etl": "ETL", "mlops": "MLOps",
    "fastapi": "FastAPI", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn", "openai": "OpenAI", "n8n": "n8n",
    "qdrant": "Qdrant", "chroma": "Chroma", "pgvector": "pgvector",
    "llamaindex": "LlamaIndex", "langchain": "LangChain",
    "hugging face": "Hugging Face", "vector database": "vector database",
    "vector db": "vector database", "graphql": "GraphQL", "go": "Go", "node": "Node",
}


def _prettify(token: str) -> str:
    if token in _CANON:
        return _CANON[token]
    if " " in token:
        return token.title()
    return token.title() if token.isalpha() else token


def extract_keywords(text: str, limit: int = 15) -> list[str]:
    """Pull skill/requirement keywords from free text, deterministically."""
    lowered = " " + text.lower() + " "
    found: list[str] = []
    seen: set[str] = set()

    for term in _MULTIWORD:
        if term in lowered and term not in seen:
            seen.add(term)
            found.append(_prettify(term))

    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9+#\-]*", text):
        token = raw.lower()
        if token in _CASE_SENSITIVE and raw != _CASE_SENSITIVE[token]:
            continue  # "go"/"rest"/"node" as ordinary words, not skills
        if token in SKILL_VOCAB and token not in seen:
            seen.add(token)
            found.append(_prettify(token))

    return found[:limit]


# Short skill names that are also ordinary English words. For grounding they only count
# when written the way a skill is written ("Go", "REST", "Node"), never "go" or "rest".
_CASE_SENSITIVE = {"go": "Go", "rest": "REST", "node": "Node"}

# Spellings that name the same skill, so "LLMs" in a resume supports "LLM" in an output.
_SAME_SKILL = {
    "llms": "llm", "apis": "api", "vector db": "vector database", "eval": "evaluation",
    "postgresql": "postgres", "agentic": "agents",
}


def skill_keys(text: str) -> set[str]:
    """Every vocabulary skill mentioned in ``text``, as canonical lower-case keys.

    Used for grounding: a skill key present in an output but absent from the source
    resume is an unsupported claim.
    """
    lowered = " " + text.lower() + " "
    keys: set[str] = set()
    for term in _MULTIWORD:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            keys.add(_SAME_SKILL.get(term, term))
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+#\-]*", text):
        low = token.lower()
        if low not in SKILL_VOCAB:
            continue
        if low in _CASE_SENSITIVE and token != _CASE_SENSITIVE[low]:
            continue
        keys.add(_SAME_SKILL.get(low, low))
    return keys


# --------------------------------------------------------------------------- #
# Tool argument contracts                                                      #
# --------------------------------------------------------------------------- #
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadResumeArgs(_Args):
    """Load the candidate's real resume text. Call this first; only use facts it contains."""


class ExtractKeywordsArgs(_Args):
    """Extract the must-have skills from a job description, split into the ones the
    candidate's resume supports and the ones it does not (never claim the latter)."""

    text: str = Field(min_length=1, max_length=20_000, description="The job description text.")


class SearchRolesArgs(_Args):
    """Find similar open roles from the local roles database, ranked by skill overlap."""

    keywords: list[str] = Field(max_length=50, description="Skills/keywords to match roles against.")
    location: Optional[str] = Field(default=None, max_length=100, description="Optional location filter.")
    top_k: int = Field(default=5, ge=1, le=20, description="How many roles to return.")


class SubmitApplicationArgs(_Args):
    """Submit the finished, tailored application. Call exactly once to finish."""

    tailored_resume_markdown: str = Field(
        min_length=1, max_length=50_000,
        description="The tailored resume in Markdown, grounded in the candidate's real resume.")
    cover_letter_markdown: str = Field(
        min_length=1, max_length=20_000,
        description="A concise, specific cover letter in Markdown (3 short paragraphs).")
    selected_role_ids: list[str] = Field(
        default_factory=list, max_length=20,
        description="IDs of the most relevant roles from search_roles.")


TOOL_ARGS: dict[str, type[_Args]] = {
    "read_resume": ReadResumeArgs,
    "extract_keywords": ExtractKeywordsArgs,
    "search_roles": SearchRolesArgs,
    "submit_application": SubmitApplicationArgs,
}


def _tool_error(kind: str, message: str) -> dict:
    return {"error": {"type": kind, "message": message}}


# --------------------------------------------------------------------------- #
# Toolbox                                                                      #
# --------------------------------------------------------------------------- #
class Toolbox:
    """Bundles the agent's tools around a session context (resume, roles, output dir).

    ``output_dir=None`` keeps submissions in memory only; the web app uses that so a
    visitor's resume is never written to the server's disk.
    """

    def __init__(self, resume_text: str, roles_db: list[dict], output_dir: Optional[str] = "outputs"):
        self.resume_text = resume_text
        self.roles_db = roles_db
        self.output_dir = output_dir
        self.last_submission: Optional[dict] = None
        self._resume_skills = skill_keys(resume_text)

    # -- individual tools ---------------------------------------------------- #
    def read_resume(self) -> dict:
        return {"resume": self.resume_text, "chars": len(self.resume_text)}

    def extract_keywords(self, text: str = "") -> dict:
        kws = extract_keywords(text or "")
        matched = [k for k in kws if skill_keys(k) & self._resume_skills]
        missing = [k for k in kws if k not in matched]
        return {"keywords": kws, "count": len(kws),
                "matched_in_resume": matched, "not_in_resume": missing}

    def search_roles(self, keywords: Optional[list[str]] = None,
                     location: Optional[str] = None, top_k: int = 5) -> dict:
        wanted = {k.lower() for k in (keywords or [])}
        scored: list[dict] = []
        for role in self.roles_db:
            tags = {t.lower() for t in role.get("tags", [])}
            if location and location.lower() not in role.get("location", "").lower():
                continue
            overlap = len(wanted & tags)
            if overlap == 0:
                continue
            union = len(wanted | tags) or 1
            scored.append({**role, "score": round(overlap / union, 3)})

        scored.sort(key=lambda r: r["score"], reverse=True)
        fallback = not scored
        if fallback:  # nothing overlaps: show a few roles, clearly marked as unranked
            scored = [{**r, "score": 0.0} for r in self.roles_db[:top_k]]

        top = scored[:top_k]
        return {"roles": top, "count": len(top), "unranked_fallback": fallback}

    def submit_application(self, tailored_resume_markdown: str, cover_letter_markdown: str,
                           selected_role_ids: Optional[list[str]] = None) -> dict:
        if self.last_submission is not None:
            return _tool_error("already_submitted", "submit_application may be called only once")
        self.last_submission = {
            "tailored_resume": tailored_resume_markdown,
            "cover_letter": cover_letter_markdown,
            "selected_role_ids": selected_role_ids or [],
        }
        saved_to = None
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            path = Path(self.output_dir) / f"application-{stamp}.md"
            path.write_text(
                f"# Tailored Resume\n\n{tailored_resume_markdown}\n\n---\n\n"
                f"# Cover Letter\n\n{cover_letter_markdown}\n",
                encoding="utf-8",
            )
            saved_to = str(path)
        return {
            "status": "submitted",
            "saved_to": saved_to,
            "selected_role_ids": selected_role_ids or [],
        }

    # -- dispatch ------------------------------------------------------------ #
    def dispatch(self, name: str, arguments: Optional[dict]) -> dict:
        """Validate and run one tool call. Never raises; failures come back as errors."""
        model = TOOL_ARGS.get(name)
        if model is None:
            return _tool_error("unknown_tool", f"no tool named {name!r}; available: {sorted(TOOL_ARGS)}")
        if arguments is None:
            return _tool_error("invalid_arguments", "arguments were not a valid JSON object")
        try:
            args = model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in exc.errors()
            )
            return _tool_error("invalid_arguments", problems)
        try:
            return getattr(self, name)(**args.model_dump())
        except Exception as exc:  # a tool bug must not crash the loop or leak a traceback
            return _tool_error("tool_failed", f"{type(exc).__name__}: {exc}")


def load_roles(path: str = "data/roles.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# OpenAI-style tool schemas, generated from the argument models               #
# --------------------------------------------------------------------------- #
def _schema(name: str, model: type[_Args]) -> dict[str, Any]:
    params = model.model_json_schema()
    params.pop("title", None)
    params.pop("description", None)
    for prop in params.get("properties", {}).values():
        prop.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": " ".join((model.__doc__ or "").split()),
            "parameters": params,
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [_schema(n, m) for n, m in TOOL_ARGS.items()]
