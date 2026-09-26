from __future__ import annotations

from src.config import settings

_client = None


def get_llm_client():
    global _client

    if _client is None:
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is missing. Set it in .env or environment variables.")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "openai package is not installed. Install requirements.txt or use mock mode."
            ) from exc
        client_options = {"api_key": settings.llm_api_key}
        if settings.llm_base_url:
            client_options["base_url"] = settings.llm_base_url
        _client = OpenAI(**client_options)

    return _client
