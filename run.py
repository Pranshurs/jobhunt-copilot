"""Command-line entry point.

Examples
--------
    # offline, zero-cost (no API key needed):
    python run.py --jd data/sample_jds/ai_engineer.txt

    # pass the JD inline:
    python run.py --jd "Senior AI Engineer at Acme. Must know RAG, LLMs, FastAPI, evaluation."

    # full JSON result (trace, roles, etc.):
    python run.py --jd data/sample_jds/ai_engineer.txt --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import JobHuntAgent
from src.config import get_settings
from src.llm import build_llm
from src.tools import Toolbox, load_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="JobHunt Copilot — tailor an application to a job description.")
    parser.add_argument("--jd", required=True, help="Path to a JD file, or the JD text itself.")
    parser.add_argument("--resume", default="data/resume.md", help="Path to the candidate resume (Markdown).")
    parser.add_argument("--roles", default="data/roles.json", help="Path to the roles database (JSON).")
    parser.add_argument("--out", default="outputs", help="Directory to save the tailored application.")
    parser.add_argument("--json", action="store_true", help="Also print the full JSON result.")
    args = parser.parse_args()

    jd_arg = Path(args.jd)
    job_description = jd_arg.read_text(encoding="utf-8") if jd_arg.exists() else args.jd
    resume = Path(args.resume).read_text(encoding="utf-8")
    roles = load_roles(args.roles)

    settings = get_settings()
    agent = JobHuntAgent(build_llm(settings), Toolbox(resume, roles, args.out), settings.max_steps)
    result = agent.run(job_description)

    print(f"\n=== JobHunt Copilot  ({settings.mode} mode) ===")
    print(f"Target      : {result.job_title}")
    print(f"Keywords    : {', '.join(result.keywords[:12]) or '-'}")
    print(f"Steps used  : {result.steps_used}")
    print(f"Matched {len(result.selected_roles)} role(s):")
    for r in result.selected_roles:
        print(f"   - {r.title} @ {r.company} ({r.location})   match={r.score}")

    print("\n--- Cover letter (preview) ---")
    print("\n".join(result.cover_letter.splitlines()[:6]))
    print(f"\nTailored application saved under: {args.out}/")

    if args.json:
        print("\n--- Full result (JSON) ---")
        print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
