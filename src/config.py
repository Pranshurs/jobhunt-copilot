"""Environment-driven settings.

The app speaks the OpenAI-compatible Chat Completions API, so the same code path
works for Groq, OpenAI, OpenRouter, Together, or a local Ollama — you only change
LLM_BASE_URL and LLM_MODEL. Leave LLM_API_KEY empty to run in offline MOCK mode
(no key, no network, no cost) — handy for demos, CI, and the eval harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional dependency; the app still runs without a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class Settings:
    mode: str          # "live" or "mock"
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_steps: int


def get_settings() -> Settings:
    """Build settings from the environment, auto-selecting mock mode if no key is set."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    mode = os.getenv("LLM_MODE", "").strip().lower()
    if mode not in {"live", "mock"}:
        mode = "live" if api_key else "mock"

    return Settings(
        mode=mode,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
    )
