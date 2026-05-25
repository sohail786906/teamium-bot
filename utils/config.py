from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if required:
            print(f"FATAL: Missing required environment variable: {name}", file=sys.stderr)
            sys.exit(1)
        return default
    return value.strip()


class Settings:
    SUPABASE_URL: str = _env("SUPABASE_URL", required=True)  # type: ignore[assignment]
    SUPABASE_SERVICE_KEY: str = _env("SUPABASE_SERVICE_KEY", required=True)  # type: ignore[assignment]

    GROQ_API_KEY: str = _env("GROQ_API_KEY", required=True)  # type: ignore[assignment]
    GROQ_BASE_URL: str = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")  # type: ignore[assignment]
    GROQ_MODEL: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")  # type: ignore[assignment]

    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # type: ignore[assignment]

    TOP_K: int = int(_env("TOP_K", "5") or "5")
    MAX_CONTEXT_CHARS: int = int(_env("MAX_CONTEXT_CHARS", "12000") or "12000")
    MAX_UPLOAD_MB: int = int(_env("MAX_UPLOAD_MB", "25") or "25")
    LLM_TIMEOUT: int = int(_env("LLM_TIMEOUT", "30") or "30")

    API_SECRET_KEY: str | None = _env("API_SECRET_KEY")


settings = Settings()
