"""Google Gemini (REST streamGenerateContent with SSE)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from app.llm.base import ChatMessage, LLMProvider, ProviderError
from app.llm.http import (
    HTTPStreamer,
    StreamTimeouts,
    UpstreamConnectionError,
    UpstreamHTTPError,
    UpstreamTimeout,
    map_upstream_error,
)

_BLOCKING_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER"}
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def to_gemini_payload(messages: list[ChatMessage], *, temperature: float, max_tokens: int, model: str) -> dict[str, Any]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "model" if m.role == "assistant" else "user"
        if contents and contents[-1]["role"] == role:  # Gemini expects alternating turns
            contents[-1]["parts"][0]["text"] += "\n\n" + m.content
        else:
            contents.append({"role": role, "parts": [{"text": m.content}]})
    generation: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if "flash" in model and "2.5" in model:
        generation["thinkingConfig"] = {"thinkingBudget": 0}  # spend the free quota on the answer, not on thinking
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation,
        # Medical textbooks discuss overdoses, self-harm, sexual health... relax the default blocking.
        "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in _SAFETY_CATEGORIES],
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


class GeminiProvider(LLMProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, *, api_key: str, model: str, http: HTTPStreamer, timeouts: StreamTimeouts | None = None):
        self.api_key = api_key
        self.model = model
        self.http = http
        self.timeouts = timeouts or StreamTimeouts()

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        body = to_gemini_payload(messages, temperature=temperature, max_tokens=max_tokens, model=self.model)
        yielded = 0
        blocked: str | None = None
        try:
            async for raw in self.http.stream_lines(url, headers=headers, json_body=body, timeouts=self.timeouts):
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except json.JSONDecodeError as exc:
                    raise ProviderError("gemini: malformed stream chunk", provider=self.name, kind="protocol") from exc
                if obj.get("error"):
                    err = obj["error"]
                    status = str(err.get("status", "")).upper()
                    kind = "rate_limit" if status == "RESOURCE_EXHAUSTED" else "unavailable"
                    raise ProviderError(
                        f"gemini: {str(err.get('message', 'stream error'))[:200]}", provider=self.name, kind=kind
                    )
                feedback = obj.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    blocked = str(feedback["blockReason"])
                for cand in obj.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        text = part.get("text")
                        if text and not part.get("thought"):
                            yielded += 1
                            yield text
                    if cand.get("finishReason") in _BLOCKING_FINISH and yielded == 0:
                        blocked = str(cand["finishReason"])
        except (UpstreamHTTPError, UpstreamTimeout, UpstreamConnectionError) as exc:
            raise map_upstream_error(self.name, exc) from exc
        if yielded == 0:
            if blocked:
                raise ProviderError(f"gemini: response blocked ({blocked})", provider=self.name, kind="blocked")
            raise ProviderError("gemini: empty response", provider=self.name, kind="empty")
