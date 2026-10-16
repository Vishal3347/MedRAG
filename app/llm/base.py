from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


# Default cooldown (seconds) a provider sits out after each kind of failure.
COOLDOWNS = {
    "rate_limit": 60.0,
    "unavailable": 20.0,
    "timeout": 15.0,
    "connection": 30.0,
    "auth": 600.0,
    "not_found": 600.0,
    "blocked": 0.0,
    "bad_request": 0.0,
    "empty": 5.0,
    "protocol": 15.0,
    "internal": 15.0,
    "error": 30.0,
}


class ProviderError(Exception):
    """Any failure of a single provider attempt. The router always responds by failing over."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        kind: str = "error",
        status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.kind = kind
        self.status = status
        self.retry_after = retry_after

    @property
    def cooldown_seconds(self) -> float:
        if self.retry_after is not None:
            return min(max(self.retry_after, 1.0), 3600.0)
        return COOLDOWNS.get(self.kind, 30.0)


class LLMProvider(ABC):
    name: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        """Async generator yielding text deltas. Raises ProviderError on any failure."""
