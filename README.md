# JobHunt Copilot

[![tests](https://github.com/Pranshurs/jobhunt-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranshurs/jobhunt-copilot/actions/workflows/ci.yml)

A small tool-calling agent that tailors a résumé and cover letter to one job description
and suggests similar roles. It comes with an evaluation harness that checks the output
only claims what the candidate's résumé supports.

**Status.** This is a small, self-contained project. There's no hosted demo; run it locally
(below). **No live-model evaluation has been run.** The
committed eval results come from the offline scripted mock, which tests the pipeline and
the harness, not model quality.

## How it works

```
job description ──► agent loop (src/agent.py) ──► LLM picks a tool ──► Toolbox.dispatch
                         ▲                                                   │
                         └─────────── tool result or typed error ◄───────────┘
tools: read_resume · extract_keywords · search_roles · submit_application (terminal, once)
```

- **Typed tool contracts.** Each tool's arguments are a pydantic model (`src/tools.py`),
  and the JSON schemas sent to the model are generated from those models. An unknown tool,
  invalid or non-JSON arguments, a second submit, or a tool that raises is returned to the
  model as `{"error": {"type", "message"}}`, so it can correct itself. None of these crash
  the loop.
- **Grounding help.** `extract_keywords` splits the job's requirements into
  `matched_in_resume` and `not_in_resume`, and the system prompt forbids claiming the
  latter.
- **Explicit end state.** Every run ends with exactly one `status`:
  - `submitted` (the only success)
  - `max_steps_exceeded`
  - `stopped_without_submitting`
  - `llm_error` (after the client's timeout and retries)
  - `invalid_input`
- **Providers.** Any OpenAI-compatible Chat Completions endpoint works: Groq, OpenAI,
  OpenRouter or local Ollama. With no `LLM_API_KEY`, it runs a deterministic offline
  **mock**. The mock is a scripted policy, not a model. It writes only skills found in both
  the job description and the résumé, and every output says which mode produced it.

## Quick start (offline, no key)

```bash
git clone https://github.com/Pranshurs/jobhunt-copilot.git && cd jobhunt-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --jd data/sample_jds/ai_engineer.txt
python run.py --jd "Data Analyst at Brightside. SQL, Python, pandas." --resume data/sample_resumes/jane_doe_data_analyst.md
```

The result is written to `outputs/`. The CLI exits 1 unless the application was submitted.

## Use a real model

```bash
cp .env.example .env   # set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL (examples inside)
python run.py --jd data/sample_jds/ai_engineer.txt
```

`LLM_TIMEOUT_S` and `LLM_MAX_RETRIES` bound each request. `AGENT_MAX_STEPS` bounds the loop.

## Web app

```bash
uvicorn app:app --reload   # http://localhost:8000: paste or upload a résumé, paste a JD
```

`POST /api/run` takes `{"job_description", "resume"?}`. It caps input sizes (422 if
exceeded), returns `status`, `mode` and `model`, and keeps the résumé in memory. Nothing is
written to the server's disk.

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest -q                                   # 47 tests: contracts, agent end states, API, harness
python -m eval.run_eval                     # the configured LLM (mock unless LLM_API_KEY is set)
python -m eval.run_eval --self-check        # known-bad policies must fail their cases
```

Each case in `eval/cases/` names a résumé, a job description and its expectations. A case
passes when all of the following hold:
- **Submitted.** The run ends `submitted`, or the case's `expected_status` if it sets one.
- **Grounded.** No vocabulary skill appears in the output that the source résumé lacks, and
  none of the case's forbidden phrases appear.
- **Covered.** Enough of the required keywords appear in the output.

There are 11 cases. Five are normal applications across three candidates; two of those
candidates are fictional samples. Six are edge or failure cases:
- a skill gap that must not be fabricated
- a prompt injection inside the job description
- another candidate's résumé, which must not leak the default one
- no skill overlap
- ordinary words that look like skills ("go", "rest")
- an empty job description

Committed results ([`eval/results/`](eval/results/)):

| Run | Passed | What it shows |
|---|---|---|
| Mock (scripted) | 11/11 | The pipeline and harness work end to end. Says nothing about model quality. |
| Self-check: `fabricating` (the old mock's behaviour) | fails 8/11, all on grounding | The grounding check catches invented skills and copied facts. |
| Self-check: `never_submit` | fails 10/11, all `max_steps_exceeded` | A run that never submits can't pass. |
| Self-check: `injection_following` | fails 1/11 (the injection case) | Obeying instructions inside a job description is caught. |

To evaluate a model, set `LLM_API_KEY` and run `python -m eval.run_eval`. The report
records the mode and model.

**Limits of the harness.** The grounding check is lexical. It only knows the skills in
`SKILL_VOCAB` (`src/tools.py`), so an invented employer or metric is caught only if a case
lists it as a forbidden phrase. A cover letter that merely *mentions* a skill the candidate
lacks also counts as a claim. Keyword coverage includes the full résumé that the mock
appends. With 11 cases, the confidence intervals are wide.

## Docker

```bash
docker build -t jobhunt-copilot . && docker run -p 8000:8000 -e LLM_API_KEY=... jobhunt-copilot
```

`.dockerignore` keeps `.env` out of the image. The image wasn't built as part of the
latest changes.

## Layout

```
run.py, app.py        CLI and FastAPI app
src/                  agent loop, tools + contracts, LLM clients (live + mock), config
eval/                 metrics, cases, runner, known-bad policies for the self-check
data/                 default résumé, fictional sample résumés, sample roles and JDs
tests/                pytest suite
```

`data/resume.md` is the author's résumé and the default profile. `data/roles.json` is
sample data.

## License

MIT
