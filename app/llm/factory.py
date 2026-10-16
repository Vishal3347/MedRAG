from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.http import HTTPStreamer, HttpxStreamer, StreamTimeouts
from app.llm.mock import MockProvider
from app.llm.ollama import OllamaProvider
from app.llm.router import LLMRouter

log = logging.getLogger(__name__)


def _secret(value: Any) -> str:
    if value is None:
        return ""
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def parse_chain(spec: str) -> list[tuple[str, str]]:
    """"groq:llama-3.3-70b-versatile, ollama:llama3.1:8b" -> [("groq", "llama-3.3-..."), ("ollama", "llama3.1:8b")]"""
    chain = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        provider, _, model = item.partition(":")
        if not model:
            raise ValueError(f"LLM_CHAIN entry '{item}' must look like provider:model")
        chain.append((provider.strip().lower(), model.strip()))
    return chain


def build_providers(settings: Any, http: HTTPStreamer) -> list[LLMProvider]:
    timeouts = StreamTimeouts(connect=settings.llm_connect_timeout_s, read=settings.llm_stall_timeout_s)
    providers: list[LLMProvider] = []
    for name, model in parse_chain(settings.llm_chain):
        if name == "groq":
            key = _secret(settings.groq_api_key)
            if not key:
                log.warning("skipping groq:%s (GROQ_API_KEY is not set)", model)
                continue
            providers.append(GroqProvider(api_key=key, model=model, http=http, timeouts=timeouts))
        elif name == "gemini":
            key = _secret(settings.gemini_api_key)
            if not key:
                log.warning("skipping gemini:%s (GEMINI_API_KEY is not set)", model)
                continue
            providers.append(GeminiProvider(api_key=key, model=model, http=http, timeouts=timeouts))
        elif name == "ollama":
            providers.append(OllamaProvider(model=model, http=http, base_url=settings.ollama_base_url, timeouts=timeouts))
        elif name == "mock":
            providers.append(MockProvider(model=model))
        else:
            raise ValueError(f"Unknown provider '{name}' in LLM_CHAIN (use groq, gemini, ollama or mock)")
    if not providers:
        log.error("no LLM providers are configured; chat requests will fail")
    return providers


def build_router(settings: Any, http: HTTPStreamer | None = None) -> tuple[LLMRouter, HTTPStreamer]:
    http = http or HttpxStreamer()
    router = LLMRouter(
        build_providers(settings, http),
        commit_chars=settings.stream_commit_chars,
        first_token_timeout=settings.llm_first_token_timeout_s,
        stall_timeout=settings.llm_stall_timeout_s,
    )
    return router, http
