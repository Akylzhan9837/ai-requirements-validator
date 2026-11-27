from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    api_key: str
    model: str
    min_interval: float
    max_retries: int
    fallback_models: list


class ConfigError(Exception):
    pass


def load_config(model_override: Optional[str] = None) -> Config:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key == "your_gemini_api_key_here":
        raise ConfigError(
            "Не задан GEMINI_API_KEY. Получите бесплатный ключ на https://aistudio.google.com/apikey, "
            "скопируйте .env.example в .env и впишите ключ (см. README).")
    model = (model_override or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    fb = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite")
    fallbacks = [m.strip() for m in fb.split(",") if m.strip() and m.strip() != model]
    return Config(
        api_key=key,
        model=model,
        fallback_models=fallbacks,
        min_interval=float(os.getenv("LLM_MIN_INTERVAL", "6")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")),
    )


def agent_version() -> str:
    """Версия агента: git-тег/коммит (+dirty, если есть незакоммиченные правки)."""
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        ref = out.stdout.strip()
        if out.returncode == 0 and ref:
            return f"{__version__} (git {ref})"
    except Exception:
        pass
    return f"{__version__} (git недоступен)"
