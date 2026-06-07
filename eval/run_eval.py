"""Run the agent across every test case and report a task-success rate.

Usage
-----
    python -m eval.run_eval                  # uses mock mode unless a key is configured
    LLM_API_KEY=... python -m eval.run_eval  # evaluate a real model

Writes ``eval/results/report.json`` and ``eval/results/report.md``.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import score_case, wilson_interval
from src.agent import JobHuntAgent
from src.config import get_settings
from src.llm import build_llm
from src.tools import Toolbox, load_roles


def load_cases(cases_dir: str = "eval/cases") -> list[dict]:
    cases = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.json"))):
        cases.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return cases


def run(cases_dir: str = "eval/cases", roles_path: str = "data/roles.json",
        out_dir: str = "eval/results") -> dict:
    settings = get_settings()
    roles = load_roles(roles_path)
    cases = load_cases(cases_dir)

    rows = []
    for case in cases:
        resume = Path(case.get("resume_path", "data/resume.md")).read_text(encoding="utf-8")
        agent = JobHuntAgent(
            build_llm(settings), Toolbox(resume, roles, os.path.join(out_dir, "_apps")),
            settings.max_steps,
        )
        result = agent.run(case["job_description"]).to_dict()
        checks = score_case(result, case.get("expectations", {}))
        rows.append({"id": case.get("id", "?"), **checks, "steps": result.get("steps_used")})

    n = len(rows)
    successes = sum(1 for r in rows if r["task_success"])
    rate = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": settings.mode,
        "model": settings.model if settings.mode == "live" else "mock",
        "n_cases": n,
        "task_success_rate": round(rate, 3),
        "task_success_ci95": [round(lo, 3), round(hi, 3)],
        "avg_keyword_coverage": round(
            statistics.mean(r["keyword_coverage"] for r in rows), 3
        ) if rows else 0.0,
        "grounding_pass_rate": round(
            sum(1 for r in rows if r["grounding_pass"]) / n, 3
        ) if n else 0.0,
        "cases": rows,
    }

    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, Path(out_dir, "report.md"))
    _print(report)
    return report


def _print(report: dict) -> None:
    bar = "=" * 66
    print("\n" + bar)
    print(f" JobHunt Copilot — Evaluation  ({report['mode']} / {report['model']})")
    print(bar)
    print(f"{'case':<30}{'success':<10}{'coverage':<11}{'ground':<9}{'struct':<8}")
    print("-" * 66)
    for r in report["cases"]:
        print(
            f"{r['id']:<30}"
            f"{('PASS' if r['task_success'] else 'FAIL'):<10}"
            f"{r['keyword_coverage']*100:>6.0f}%    "
            f"{('ok' if r['grounding_pass'] else 'X'):<9}"
            f"{('ok' if r['structure_pass'] else 'X'):<8}"
        )
    print("-" * 66)
    lo, hi = report["task_success_ci95"]
    print(
        f" Task-success rate: {report['task_success_rate']*100:.0f}%"
        f"  (95% CI {lo*100:.0f}–{hi*100:.0f}%, n={report['n_cases']})"
    )
    print(
        f" Avg keyword coverage: {report['avg_keyword_coverage']*100:.0f}%"
        f"   |   Grounding pass rate: {report['grounding_pass_rate']*100:.0f}%"
    )
    print(bar + "\n")


def _write_markdown(report: dict, path: Path) -> None:
    lo, hi = report["task_success_ci95"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Mode: **{report['mode']}** (`{report['model']}`)",
        f"- Cases: **{report['n_cases']}**",
        f"- **Task-success rate: {report['task_success_rate']*100:.0f}%** "
        f"(95% CI {lo*100:.0f}–{hi*100:.0f}%)",
        f"- Avg keyword coverage: **{report['avg_keyword_coverage']*100:.0f}%**",
        f"- Grounding pass rate: **{report['grounding_pass_rate']*100:.0f}%**",
        "",
        "| Case | Success | Coverage | Grounded | Structured |",
        "|------|---------|----------|----------|------------|",
    ]
    for r in report["cases"]:
        lines.append(
            f"| `{r['id']}` | {'✅' if r['task_success'] else '❌'} | "
            f"{r['keyword_coverage']*100:.0f}% | "
            f"{'✅' if r['grounding_pass'] else '❌'} | "
            f"{'✅' if r['structure_pass'] else '❌'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
