"""Offline providers: an extractive demo model and a scriptable failure injector for tests."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from app.llm.base import ChatMessage, LLMProvider, ProviderError


class MockProvider(LLMProvider):
    """`LLM_CHAIN=mock:echo` lets you run the whole stack end-to-end with no API keys. It answers
    by quoting the first retrieved passage, which also makes citation rendering easy to check."""

    name = "mock"

    def __init__(self, model: str = "echo", delay_s: float = 0.0):
        self.model = model
        self.delay_s = delay_s

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        match = re.search(r"\[S1\][^\n]*\n(.+?)(?:\n\n|\Z)", user, re.S)
        excerpt = " ".join(match.group(1).split())[:400] if match else "no passage was provided"
        text = f"(mock model) The first retrieved passage says: {excerpt} [S1]"
        for word in text.split(" "):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield word + " "


@dataclass(frozen=True, slots=True)
class Fail:
    """Script step: raise a ProviderError."""

    kind: str = "unavailable"
    status: int | None = None
    retry_after: float | None = None


@dataclass(frozen=True, slots=True)
class Sleep:
    """Script step: stall (used to test timeouts)."""

    seconds: float


Step = str | Fail | Sleep


class ScriptedProvider(LLMProvider):
    """Plays back one script per call (the last script repeats). A script is a list of steps:
    a string is a token, Fail raises ProviderError, Sleep waits."""

    def __init__(self, name: str, model: str, scripts: Sequence[Sequence[Step]]):
        self.name = name
        self.model = model
        self._scripts = [list(s) for s in scripts]
        self.calls = 0

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        for step in script:
            if isinstance(step, Fail):
                raise ProviderError(
                    f"{self.name}: scripted failure", provider=self.name, kind=step.kind, status=step.status,
                    retry_after=step.retry_after,
                )
            if isinstance(step, Sleep):
                await asyncio.sleep(step.seconds)
            else:
                yield step
