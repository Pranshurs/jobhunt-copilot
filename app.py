"""FastAPI server exposing the JobHunt Copilot agent — this is the live demo.

Run locally:
    uvicorn app:app --reload
    # open http://localhost:8000

It serves a single-page UI at "/" and a JSON API at "/api/run".
Works in offline mock mode out of the box; set LLM_API_KEY for live mode.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agent import JobHuntAgent
from src.config import get_settings
from src.llm import build_llm
from src.tools import Toolbox, load_roles

ROOT = Path(__file__).parent
RESUME = (ROOT / "data" / "resume.md").read_text(encoding="utf-8")
ROLES = load_roles(str(ROOT / "data" / "roles.json"))

app = FastAPI(title="JobHunt Copilot", version="0.1.0")


class RunRequest(BaseModel):
    job_description: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": get_settings().mode}


@app.post("/api/run")
def run(req: RunRequest) -> JSONResponse:
    settings = get_settings()
    agent = JobHuntAgent(
        build_llm(settings),
        Toolbox(RESUME, ROLES, str(ROOT / "outputs")),
        settings.max_steps,
    )
    result = agent.run(req.job_description)
    return JSONResponse(result.to_dict())
