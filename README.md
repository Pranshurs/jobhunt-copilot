# JobHunt Copilot 🎯

**An AI agent that tailors your job application — and an evaluation harness that proves it works.**

Most "AI agent" demos look impressive and quietly hallucinate. This project is the opposite: a tool-using agent that tailors a resume, drafts a cover letter, and finds matching roles — wrapped in an **evaluation harness that scores task success, keyword coverage, and grounding** across a suite of test cases, with a Wilson confidence interval on the result.

> Building agents is common. *Measuring* whether they work is rare. This repo does both.

- 🤖 **Agentic** — a real tool-calling loop (read → analyse → search → submit)
- 📊 **Evaluated** — automated task-success scoring + hallucination/grounding checks
- 🆓 **Runs with zero setup** — offline mock mode means no API key, no cost, no network
- 🔌 **Provider-agnostic** — Groq, OpenAI, OpenRouter, or local Ollama via one env var
- 🚀 **Deployable** — FastAPI web demo + Dockerfile, ready for a VPS

---

## Why this exists

I build AI systems and I evaluate them. I ran LLM reasoning evaluation at Outlier and
built statistical rigour into a full ML research platform. This project turns that edge
into something you can run: an agent whose quality is **measured, not asserted.**

---

## Architecture

```
            ┌──────────────────────────────────────────────────────────┐
            │                      JobHunt Agent                        │
   Job      │                                                          │
Description ─┼──►  LLM loop  ──►  decides which tool to call next        │
 + Resume   │        ▲                     │                            │
            │        │                     ▼                            │
            │        │      ┌───────────────────────────────┐           │
            │        └──────┤ Tools                          │           │
            │  tool results │  • read_resume                 │           │
            │               │  • extract_keywords            │           │
            │               │  • search_roles (retrieval)    │           │
            │               │  • submit_application (action) │           │
            │               └───────────────────────────────┘           │
            └───────────────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                        Tailored resume · Cover letter · Role matches
                                         │
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
            │  Evaluation harness  (the differentiator)                 │
            │  test cases ─► run agent ─► score:                        │
            │     • keyword coverage   • grounding / hallucination      │
            │     • structure          ─► task-success rate ± 95% CI    │
            └──────────────────────────────────────────────────────────┘
```

The agent and the eval share the same code path, so the harness measures *exactly* what
ships. Swapping the LLM provider (or running fully offline) changes nothing about how it's scored.

---

## Quickstart (no API key needed)

```bash
git clone https://github.com/Pranshurs/jobhunt-copilot.git && cd jobhunt-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the agent on a sample job description (offline mock mode):
python run.py --jd data/sample_jds/ai_engineer.txt
```

You'll get a tailored resume, a grounded cover letter, and ranked role matches — saved to `outputs/`.

### Web demo

```bash
uvicorn app:app --reload
# open http://localhost:8000  → paste a JD → "Tailor my application"
```

### Live LLM mode

```bash
cp .env.example .env          # then set LLM_API_KEY (Groq has a free tier)
python run.py --jd data/sample_jds/ai_engineer.txt
```

The app speaks the OpenAI-compatible API, so the same code runs against **Groq, OpenAI,
OpenRouter, Together, or a local Ollama** — you only change `LLM_BASE_URL` and `LLM_MODEL`.

---

## Evaluation

```bash
python -m eval.run_eval
```

Sample run (offline mock mode, 3 cases):

```
==================================================================
 JobHunt Copilot — Evaluation  (mock / mock)
==================================================================
case                          success   coverage   ground   struct
------------------------------------------------------------------
ai_engineer_rag               PASS         100%    ok       ok
applied_ai_agents             PASS         100%    ok       ok
ai_automation_engineer        PASS         100%    ok       ok
------------------------------------------------------------------
 Task-success rate: 100%  (95% CI 44–100%, n=3)
 Avg keyword coverage: 100%   |   Grounding pass rate: 100%
==================================================================
```

Each case (`eval/cases/*.json`) declares its own expectations, and the harness scores three things:

| Metric | What it checks | How |
|---|---|---|
| **Keyword coverage** | Did the application surface the job's key requirements? | fraction of required keywords present, vs. a threshold |
| **Grounding** | Did it avoid claims the candidate can't back up? | none of the `forbidden_claims` appear (hallucination guard) |
| **Structure** | Is the deliverable complete? | tailored resume + cover letter present, ≥ N roles matched |

`task_success` is the AND of all three. The reported success rate ships with a **Wilson
score 95% confidence interval** — because a point estimate over a handful of cases without
an interval is misleading (note how wide `44–100%` correctly is at n=3). Results are written
to `eval/results/report.json` and `report.md`.

---

## How the agent works

The loop (`src/agent.py`) hands the LLM a set of tools (`src/tools.py`) and lets it decide
what to call, in OpenAI function-calling format:

1. `read_resume` — load the candidate's real resume (ground truth)
2. `extract_keywords` — pull must-have skills from the job description
3. `search_roles` — retrieve & rank similar open roles by skill overlap
4. `submit_application` — the terminal action: tailored resume + cover letter, persisted to disk

Every step is recorded in a trace for transparency. The offline `MockLLM` deterministically
simulates this exact sequence, which is what makes the project runnable — and testable in CI —
with no key or cost.

---

## Project structure

```
jobhunt-copilot/
├── run.py                  # CLI entry point
├── app.py                  # FastAPI server (the live demo)
├── templates/index.html    # single-page demo UI
├── src/
│   ├── config.py           # env-driven settings (auto mock/live)
│   ├── schemas.py          # typed data structures
│   ├── llm.py              # LiveLLM (OpenAI-compatible) + offline MockLLM
│   ├── tools.py            # the agent's tools + OpenAI tool schemas
│   ├── agent.py            # the tool-calling loop
│   └── prompts.py          # system prompt
├── eval/
│   ├── metrics.py          # coverage, grounding, structure, Wilson CI
│   ├── run_eval.py         # runner + JSON/Markdown report
│   └── cases/*.json        # test cases with expectations
├── data/
│   ├── resume.md           # candidate profile (replace with your own)
│   ├── roles.json          # sample roles database
│   └── sample_jds/*.txt    # example job descriptions
├── tests/                  # pytest suite (tools, metrics, agent e2e)
├── Dockerfile
└── requirements.txt
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q          # 12 tests: tools, metrics, and an end-to-end agent run
```

---

## Deployment (Docker on a VPS)

```bash
docker build -t jobhunt-copilot .
docker run -d --name jobhunt -p 8000:8000 \
  -e LLM_API_KEY=your_key -e LLM_BASE_URL=https://api.groq.com/openai/v1 \
  -e LLM_MODEL=llama-3.3-70b-versatile \
  jobhunt-copilot
```

Put it behind nginx (or Caddy) with TLS and point a subdomain at it.
**Live demo:** `https://<your-domain>` *(add once deployed)*

---

## Roadmap

This is project 1 of a small, focused portfolio:

1. **JobHunt Copilot** — agent + evaluation *(this repo)*
2. **RAG that measures itself** — retrieval-quality scoring + citation/hallucination checks
3. **LLM evaluation toolkit** — run prompts across models, score reasoning/math/grounding

---

## Notes

`data/resume.md` is a sample profile used as the agent's default candidate — replace it with
your own resume text. The roles database is sample data; swapping in a real job-board feed
(or a live search tool) is a natural next step.

## Author

**Pranshu Raj** — AI/ML Engineer, Gurgaon, India
GitHub: `github.com/Pranshurs` · Email: pranshu.rs08@gmail.com
