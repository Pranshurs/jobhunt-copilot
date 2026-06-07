"""Scoring metrics for the agent's output.

The harness measures three things per test case and combines them into a single
pass/fail ``task_success`` flag:

* keyword_coverage — did the tailored application surface the job's key requirements?
* grounding        — did it avoid claims the candidate can't back up (hallucination guard)?
* structure        — are all required sections present and is at least one role matched?

It also reports a Wilson score 95% confidence interval on the success rate, because a
point estimate over a handful of cases without an interval is misleading.
"""

from __future__ import annotations

import math


def keyword_coverage(text: str, required: list[str]) -> float:
    """Fraction of required keywords present in the text (case-insensitive substring)."""
    if not required:
        return 1.0
    blob = text.lower()
    hits = sum(1 for k in required if k.lower() in blob)
    return hits / len(required)


def grounding_ok(texts: list[str], forbidden: list[str]) -> bool:
    """True if none of the forbidden (unsupported) claims appear anywhere in the texts."""
    blob = " ".join(texts).lower()
    return not any(f.lower() in blob for f in (forbidden or []))


def structure_complete(result: dict, must_include: list[str], min_roles: int) -> bool:
    """True if every required section is present/non-empty and enough roles were matched."""
    for section in must_include:
        if section == "role_matches":
            continue  # checked via min_roles below
        if not str(result.get(section, "")).strip():
            return False
    return len(result.get("selected_roles", [])) >= min_roles


def score_case(result: dict, expectations: dict) -> dict:
    """Score a single agent result against a case's expectations."""
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
    grounded = grounding_ok([tailored, cover], forbidden)
    structured = structure_complete(result, must_include, min_roles)

    checks = {
        "keyword_coverage": round(coverage, 3),
        "coverage_pass": coverage >= coverage_threshold,
        "grounding_pass": grounded,
        "structure_pass": structured,
    }
    checks["task_success"] = bool(checks["coverage_pass"] and grounded and structured)
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
