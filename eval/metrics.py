"""Scoring metrics for the agent's output.

The harness measures three things per test case and combines them into a single
pass/fail ``task_success`` flag:

* keyword_coverage — did the tailored application surface the job's key requirements?
* grounding        — no skill named in the output that the source resume lacks, and none
                     of the case's forbidden phrases (fabricated degrees, employers, ...)
* structure        — the run submitted, all required sections are present, enough roles

A case can instead expect a non-success end state (``expected_status``, e.g. an empty job
description must end ``invalid_input``); it then passes only if the run ends that way.

The skill check is lexical: it looks for vocabulary skills (see src/tools.py) in the
tailored resume and the cover letter. A letter that *mentions* a skill the candidate lacks
("your stack uses Kubernetes") counts as an unsupported claim, so the check errs strict.

It also reports a Wilson score 95% confidence interval on the success rate, because a
point estimate over a handful of cases without an interval is misleading.
"""

from __future__ import annotations

import math

from src.tools import skill_keys


def keyword_coverage(text: str, required: list[str]) -> float:
    """Fraction of required keywords present in the text (case-insensitive substring)."""
    if not required:
        return 1.0
    blob = text.lower()
    hits = sum(1 for k in required if k.lower() in blob)
    return hits / len(required)


def grounding_ok(texts: list[str], forbidden: list[str]) -> bool:
    """True if none of the forbidden (unsupported) claims appear anywhere in the texts."""
    return not forbidden_hits(texts, forbidden)


def forbidden_hits(texts: list[str], forbidden: list[str]) -> list[str]:
    blob = " ".join(texts).lower()
    return [f for f in (forbidden or []) if f.lower() in blob]


def unsupported_skills(texts: list[str], resume: str) -> list[str]:
    """Vocabulary skills named in the output that the source resume doesn't contain."""
    claimed = set().union(*(skill_keys(t) for t in texts)) if texts else set()
    return sorted(claimed - skill_keys(resume))


def structure_complete(result: dict, must_include: list[str], min_roles: int) -> bool:
    """True if every required section is present/non-empty and enough roles were matched."""
    for section in must_include:
        if section == "role_matches":
            continue  # checked via min_roles below
        if not str(result.get(section, "")).strip():
            return False
    return len(result.get("selected_roles", [])) >= min_roles


def score_case(result: dict, expectations: dict, resume: str = "") -> dict:
    """Score a single agent result against a case's expectations.

    ``resume`` is the source resume the run was given; without it the skill-grounding
    check is skipped (kept for callers that only have the output).
    """
    status = result.get("status", "submitted")
    expected_status = expectations.get("expected_status", "submitted")
    if expected_status != "submitted":
        ok = status == expected_status
        return {"status": status, "keyword_coverage": None, "coverage_pass": ok,
                "grounding_pass": ok, "structure_pass": ok, "unsupported_skills": [],
                "forbidden_hits": [], "task_success": ok}

    required = expectations.get("required_keywords", [])
    forbidden = expectations.get("forbidden_claims", [])
    must_include = expectations.get(
        "must_include_sections", ["tailored_resume", "cover_letter", "role_matches"]
    )
    min_roles = int(expectations.get("min_role_matches", 1))
    coverage_threshold = float(expectations.get("coverage_threshold", 0.7))

    tailored = result.get("tailored_resume", "") or ""
    cover = result.get("cover_letter", "") or ""

    coverage = keyword_coverage(f"{tailored}\n{cover}", required)
    hits = forbidden_hits([tailored, cover], forbidden)
    unsupported = unsupported_skills([tailored, cover], resume) if resume else []
    structured = status == "submitted" and structure_complete(result, must_include, min_roles)

    checks = {
        "status": status,
        "keyword_coverage": round(coverage, 3),
        "coverage_pass": coverage >= coverage_threshold,
        "grounding_pass": not hits and not unsupported,
        "structure_pass": structured,
        "unsupported_skills": unsupported,
        "forbidden_hits": hits,
    }
    checks["task_success"] = bool(checks["coverage_pass"] and checks["grounding_pass"] and structured)
    return checks


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
