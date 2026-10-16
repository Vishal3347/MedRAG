"""Groq (OpenAI-compatible chat completions with SSE streaming)."""
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


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"
    base_url = ""

    def __init__(self, *, api_key: str, model: str, http: HTTPStreamer, timeouts: StreamTimeouts | None = None):
        self.api_key = api_key
        self.model = model
        self.http = http
        self.timeouts = timeouts or StreamTimeouts()

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.model.startswith("openai/gpt-oss"):
            # Reasoning model: keep thinking short so it does not eat the max_tokens budget, and don't stream it.
            body["reasoning_effort"] = "low"
            body["include_reasoning"] = False
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        yielded = 0
        try:
            async for raw in self.http.stream_lines(
                f"{self.base_url}/chat/completions", headers=headers, json_body=body, timeouts=self.timeouts
            ):
                line = raw.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"{self.name}: malformed stream chunk", provider=self.name, kind="protocol") from exc
                if obj.get("error"):
                    err = obj["error"] if isinstance(obj["error"], dict) else {"message": str(obj["error"])}
                    marker = f"{err.get('type', '')} {err.get('code', '')}".lower()
                    kind = "rate_limit" if "rate_limit" in marker else "unavailable"
                    raise ProviderError(
                        f"{self.name}: {str(err.get('message', 'stream error'))[:200]}", provider=self.name, kind=kind
                    )
                for choice in obj.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yielded += 1
                        yield text
        except (UpstreamHTTPError, UpstreamTimeout, UpstreamConnectionError) as exc:
            raise map_upstream_error(self.name, exc) from exc
        if yielded == 0:
            raise ProviderError(f"{self.name}: empty response", provider=self.name, kind="empty")


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
