"""Agent tools and their OpenAI-style schemas.

The tools are deliberately concrete and deterministic (resume I/O, keyword
extraction, role retrieval/ranking, and a terminal "submit" action). The agent
LLM decides *when* to call them and composes the final deliverables; the tools
do the grunt work. This split keeps the agent testable and the behaviour legible.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    "vector db": "vector database", "graphql": "GraphQL",
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

    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#\-]*", text.lower()):
        if token in SKILL_VOCAB and token not in seen:
            seen.add(token)
            found.append(_prettify(token))

    return found[:limit]


# --------------------------------------------------------------------------- #
# Toolbox                                                                      #
# --------------------------------------------------------------------------- #
class Toolbox:
    """Bundles the agent's tools around a session context (resume, roles, output dir)."""

    def __init__(self, resume_text: str, roles_db: list[dict], output_dir: str = "outputs"):
        self.resume_text = resume_text
        self.roles_db = roles_db
        self.output_dir = output_dir
        self.last_submission: Optional[dict] = None

    # -- individual tools ---------------------------------------------------- #
    def read_resume(self) -> dict:
        return {"resume": self.resume_text, "chars": len(self.resume_text)}

    def extract_keywords(self, text: str = "") -> dict:
        kws = extract_keywords(text or "")
        return {"keywords": kws, "count": len(kws)}

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
        if not scored:  # graceful fallback so the agent always has something to show
            scored = [{**r, "score": 0.0} for r in self.roles_db[:top_k]]

        top = scored[:top_k]
        return {"roles": top, "count": len(top)}

    def submit_application(self, tailored_resume_markdown: str, cover_letter_markdown: str,
                           selected_role_ids: Optional[list[str]] = None) -> dict:
        self.last_submission = {
            "tailored_resume": tailored_resume_markdown,
            "cover_letter": cover_letter_markdown,
            "selected_role_ids": selected_role_ids or [],
        }
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = Path(self.output_dir) / f"application-{stamp}.md"
        path.write_text(
            f"# Tailored Resume\n\n{tailored_resume_markdown}\n\n---\n\n"
            f"# Cover Letter\n\n{cover_letter_markdown}\n",
            encoding="utf-8",
        )
        return {
            "status": "submitted",
            "saved_to": str(path),
            "selected_role_ids": selected_role_ids or [],
        }

    # -- dispatch ------------------------------------------------------------ #
    def dispatch(self, name: str, arguments: dict) -> dict:
        args = arguments or {}
        if name == "read_resume":
            return self.read_resume()
        if name == "extract_keywords":
            return self.extract_keywords(args.get("text", ""))
        if name == "search_roles":
            return self.search_roles(
                args.get("keywords"), args.get("location"), int(args.get("top_k", 5) or 5)
            )
        if name == "submit_application":
            return self.submit_application(
                args.get("tailored_resume_markdown", ""),
                args.get("cover_letter_markdown", ""),
                args.get("selected_role_ids", []),
            )
        return {"error": f"unknown tool: {name}"}


def load_roles(path: str = "data/roles.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# OpenAI-style tool schemas (used in live mode for function calling)          #
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_resume",
            "description": "Load the candidate's real resume text. Call this first; only use facts it contains.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "Extract the must-have skills and requirements from a job description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The job description text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_roles",
            "description": "Find similar open roles from the local roles database, ranked by skill overlap.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills/keywords to match roles against.",
                    },
                    "location": {"type": "string", "description": "Optional location filter."},
                    "top_k": {"type": "integer", "description": "How many roles to return."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_application",
            "description": "Submit the finished, tailored application. Call exactly once to finish.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tailored_resume_markdown": {
                        "type": "string",
                        "description": "The tailored resume in Markdown, grounded in the candidate's real resume.",
                    },
                    "cover_letter_markdown": {
                        "type": "string",
                        "description": "A concise, specific cover letter in Markdown (3 short paragraphs).",
                    },
                    "selected_role_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of the most relevant roles from search_roles.",
                    },
                },
                "required": ["tailored_resume_markdown", "cover_letter_markdown"],
            },
        },
    },
]
