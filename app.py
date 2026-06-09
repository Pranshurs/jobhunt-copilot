"""FastAPI server exposing the JobHunt Copilot agent — this is the live demo.

Run locally:
    uvicorn app:app --reload
    # open http://localhost:8000

Serves a single-page UI at "/" and a JSON API:
    POST /api/run      {job_description, resume?}  -> tailored application
    POST /api/extract  (multipart file)            -> plain text from PDF/DOCX/TXT
    GET  /api/health
Works in offline mock mode out of the box; set LLM_API_KEY for live mode.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.agent import JobHuntAgent
from src.config import get_settings
from src.llm import build_llm
from src.tools import Toolbox, load_roles

ROOT = Path(__file__).parent
DEFAULT_RESUME = (ROOT / "data" / "resume.md").read_text(encoding="utf-8")
ROLES = load_roles(str(ROOT / "data" / "roles.json"))

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB cap on an uploaded résumé
ALLOWED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")

app = FastAPI(title="JobHunt Copilot", version="0.2.0")


class RunRequest(BaseModel):
    job_description: str
    resume: Optional[str] = None  # if omitted, falls back to the demo résumé


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": get_settings().mode}


@app.post("/api/run")
def run(req: RunRequest) -> JSONResponse:
    settings = get_settings()
    resume = (req.resume or "").strip() or DEFAULT_RESUME
    agent = JobHuntAgent(
        build_llm(settings),
        Toolbox(resume, ROLES, str(ROOT / "outputs")),
        settings.max_steps,
    )
    result = agent.run(req.job_description)
    return JSONResponse(result.to_dict())


def _extract_text(filename: str, data: bytes) -> str:
    """Best-effort plain-text extraction from a résumé file. Lazy imports keep
    the heavy parsers optional until someone actually uploads that format."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        import docx

        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    # .txt / .md / plain text
    return data.decode("utf-8", errors="ignore")


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    name = (file.filename or "").lower()
    if not name.endswith(ALLOWED_SUFFIXES):
        return JSONResponse(
            {"error": "Unsupported file type. Use PDF, DOCX, TXT, or MD — or just paste the text."},
            status_code=415,
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File too large (max 2 MB)."}, status_code=413)
    try:
        text = _extract_text(file.filename or "", data).strip()
    except Exception:
        return JSONResponse(
            {"error": "Couldn't read that file. Try pasting the text instead."},
            status_code=422,
        )
    return JSONResponse({"text": text[:20000], "filename": file.filename})
