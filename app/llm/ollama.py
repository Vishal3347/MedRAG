"""Ollama local fallback (NDJSON streaming from /api/chat)."""
from __future__ import annotations

import json
from typing import AsyncIterator

from app.llm.base import ChatMessage, LLMProvider, ProviderError
from app.llm.http import (
    HTTPStreamer,
    StreamTimeouts,
    UpstreamConnectionError,
    UpstreamHTTPError,
    UpstreamTimeout,
    map_upstream_error,
)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self, *, model: str, http: HTTPStreamer, base_url: str = "http://localhost:11434",
        timeouts: StreamTimeouts | None = None,
    ):
        self.model = model
        self.http = http
        self.base_url = base_url.rstrip("/")
        self.timeouts = timeouts or StreamTimeouts()

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        yielded = 0
        try:
            async for raw in self.http.stream_lines(
                f"{self.base_url}/api/chat", headers={"Content-Type": "application/json"}, json_body=body,
                timeouts=self.timeouts,
            ):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProviderError("ollama: malformed stream chunk", provider=self.name, kind="protocol") from exc
                if obj.get("error"):
                    raise ProviderError(f"ollama: {str(obj['error'])[:200]}", provider=self.name, kind="unavailable")
                text = (obj.get("message") or {}).get("content")
                if text:
                    yielded += 1
                    yield text
                if obj.get("done"):
                    break
        except (UpstreamHTTPError, UpstreamTimeout, UpstreamConnectionError) as exc:
            raise map_upstream_error(self.name, exc) from exc
        if yielded == 0:
            raise ProviderError("ollama: empty response", provider=self.name, kind="empty")
