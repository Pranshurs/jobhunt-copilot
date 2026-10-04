"""Run the agent across every test case and report raw per-case results.

Usage
-----
    python -m eval.run_eval                     # the configured LLM: mock unless LLM_API_KEY is set
    python -m eval.run_eval --policy mock       # force the scripted mock
    python -m eval.run_eval --self-check        # prove the metrics fail on known-bad policies
    python -m eval.run_eval --require-all-pass  # exit 1 if any case fails (used in CI)

Writes ``report.json`` / ``report.md`` (and ``self_check.json``) to ``--out-dir``
(default ``eval/results``).

Mock-mode numbers measure the pipeline and the harness, not model quality: the mock is a
scripted policy. Only a run with ``LLM_API_KEY`` set evaluates a model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from eval.metrics import score_case, wilson_interval
from eval.policies import BAD_POLICIES
from src.agent import JobHuntAgent
from src.config import Settings, get_settings
from src.llm import BaseLLM, MockLLM, build_llm
from src.tools import Toolbox, load_roles

# Which cases each known-bad policy must fail, and the check that must catch it. The
# self-check fails if a policy fails anything else, or passes any of these. The
# fabricating policy passes the cases where every requested skill is genuinely on the
# default resume: there is nothing to fabricate there, so passing them is correct.
SELF_CHECK_EXPECTED = {
    "fabricating": {
        "check": "grounding_pass",
        "must_fail": [
            "ai_automation_engineer", "ai_engineer_rag", "data_analyst_matching_resume",
            "edge_ambiguous_words_not_skills", "edge_no_skill_overlap",
            "edge_other_candidate_no_author_leak", "edge_skill_gap_must_not_fabricate",
            "platform_engineer_matching_resume",
        ],
    },
    "never_submit": {
        "check": "status",
        "must_fail": [
            "ai_automation_engineer", "ai_engineer_rag", "applied_ai_agents",
            "data_analyst_matching_resume", "edge_ambiguous_words_not_skills",
            "edge_no_skill_overlap", "edge_other_candidate_no_author_leak",
            "edge_prompt_injection_in_jd", "edge_skill_gap_must_not_fabricate",
            "platform_engineer_matching_resume",
        ],
    },
    "injection_following": {
        "check": "grounding_pass",
        "must_fail": ["edge_prompt_injection_in_jd"],
    },
}


def load_cases(cases_dir: str = "eval/cases") -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(cases_dir, "*.json")))]


def evaluate(make_llm: Callable[[], BaseLLM], cases: list[dict], roles: list[dict],
             max_steps: int) -> list[dict]:
    rows = []
    for case in cases:
        resume = Path(case.get("resume_path", "data/resume.md")).read_text(encoding="utf-8")
        agent = JobHuntAgent(make_llm(), Toolbox(resume, roles, output_dir=None), max_steps)
        result = agent.run(case["job_description"]).to_dict()
        checks = score_case(result, case.get("expectations", {}), resume=resume)
        rows.append({"id": case.get("id", "?"), **checks, "steps": result.get("steps_used"),
                     "tool_errors": sum(1 for t in result["trace"] if not t["ok"])})
    return rows


def summarize(rows: list[dict], mode: str, model: str) -> dict:
    n = len(rows)
    successes = sum(1 for r in rows if r["task_success"])
    lo, hi = wilson_interval(successes, n)
    covs = [r["keyword_coverage"] for r in rows if r["keyword_coverage"] is not None]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": model,
        "n_cases": n,
        "n_passed": successes,
        "task_success_rate": round(successes / n, 3) if n else 0.0,
        "task_success_ci95": [round(lo, 3), round(hi, 3)],
        "avg_keyword_coverage": round(statistics.mean(covs), 3) if covs else None,
        "grounding_failures": sum(1 for r in rows if not r["grounding_pass"]),
        "cases": rows,
    }


def self_check(cases: list[dict], roles: list[dict], settings: Settings) -> dict:
    """Run each known-bad policy and compare its failures with SELF_CHECK_EXPECTED."""
    out = {}
    for name, cls in BAD_POLICIES.items():
        rows = evaluate(lambda cls=cls: cls(settings), cases, roles, settings.max_steps)
        spec = SELF_CHECK_EXPECTED[name]
        failed = sorted(r["id"] for r in rows if not r["task_success"])
        if spec["check"] == "status":
            wrong_reason = [r["id"] for r in rows if not r["task_success"] and r["status"] != "max_steps_exceeded"]
        else:
            wrong_reason = [r["id"] for r in rows if not r["task_success"] and r[spec["check"]]]
        out[name] = {
            "failed": failed,
            "expected": sorted(spec["must_fail"]),
            "missed": sorted(set(spec["must_fail"]) - set(failed)),
            "unexpected": sorted(set(failed) - set(spec["must_fail"])),
            "failed_for_another_reason": wrong_reason,
        }
        out[name]["ok"] = not (out[name]["missed"] or out[name]["unexpected"] or wrong_reason)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=["configured", "mock", *BAD_POLICIES], default="configured")
    ap.add_argument("--cases", default="eval/cases")
    ap.add_argument("--roles", default="data/roles.json")
    ap.add_argument("--out-dir", default="eval/results")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--require-all-pass", action="store_true")
    args = ap.parse_args(argv)

    settings = get_settings()
    cases, roles = load_cases(args.cases), load_roles(args.roles)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.self_check:
        res = self_check(cases, roles, settings)
        Path(args.out_dir, "self_check.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        for name, r in res.items():
            print(f"self-check {name:<20} {'OK ' if r['ok'] else 'BAD'} failed {len(r['failed'])}/{len(cases)}"
                  f" (expected {len(r['expected'])}; missed {r['missed']}; unexpected {r['unexpected']};"
                  f" wrong reason {r['failed_for_another_reason']})")
        return 0 if all(r["ok"] for r in res.values()) else 1

    if args.policy == "configured":
        make_llm, mode = (lambda: build_llm(settings)), settings.mode
    elif args.policy == "mock":
        make_llm, mode = (lambda: MockLLM(settings)), "mock"
    else:
        make_llm, mode = (lambda: BAD_POLICIES[args.policy](settings)), f"bad-policy:{args.policy}"
    model = settings.model if mode == "live" else "scripted policy, not a model"

    report = summarize(evaluate(make_llm, cases, roles, settings.max_steps), mode, model)
    Path(args.out_dir, "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, Path(args.out_dir, "report.md"))
    _print(report)
    return 1 if args.require_all_pass and report["n_passed"] != report["n_cases"] else 0


def _fmt_cov(c) -> str:
    return "  n/a" if c is None else f"{c*100:4.0f}%"


def _print(report: dict) -> None:
    bar = "=" * 92
    print("\n" + bar)
    print(f" JobHunt Copilot evaluation  ({report['mode']} / {report['model']})")
    print(bar)
    print(f"{'case':<40}{'result':<8}{'status':<20}{'cov':<7}{'grounded':<10}unsupported")
    print("-" * 92)
    for r in report["cases"]:
        print(f"{r['id']:<40}{('PASS' if r['task_success'] else 'FAIL'):<8}{r['status']:<20}"
              f"{_fmt_cov(r['keyword_coverage']):<7}{('ok' if r['grounding_pass'] else 'X'):<10}"
              f"{', '.join(r['unsupported_skills'] + r['forbidden_hits']) or '-'}")
    print("-" * 92)
    lo, hi = report["task_success_ci95"]
    print(f" Passed {report['n_passed']}/{report['n_cases']}  (Wilson 95% CI {lo*100:.0f}–{hi*100:.0f}%)"
          f"  | grounding failures: {report['grounding_failures']}")
    if report["mode"] == "mock":
        print(" Mock mode: this checks the pipeline and the harness, not a model.")
    print(bar + "\n")


def _write_markdown(report: dict, path: Path) -> None:
    lo, hi = report["task_success_ci95"]
    lines = [
        "# Evaluation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Mode: **{report['mode']}** ({report['model']})",
        f"- Passed: **{report['n_passed']}/{report['n_cases']}** (Wilson 95% CI {lo*100:.0f}–{hi*100:.0f}%)",
        f"- Grounding failures: {report['grounding_failures']}",
    ]
    if report["mode"] == "mock":
        lines.append("- Mock mode checks the pipeline and the harness, not a model.")
    lines += ["", "| Case | Result | Status | Coverage | Grounded | Unsupported / forbidden |",
              "|---|---|---|---|---|---|"]
    for r in report["cases"]:
        lines.append(
            f"| `{r['id']}` | {'pass' if r['task_success'] else 'FAIL'} | {r['status']} | "
            f"{_fmt_cov(r['keyword_coverage']).strip()} | {'yes' if r['grounding_pass'] else 'no'} | "
            f"{', '.join(r['unsupported_skills'] + r['forbidden_hits']) or '-'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
