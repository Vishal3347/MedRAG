"""Provider-independent LLM router with in-request failover and a pre-commit token buffer.

Streaming contract (events yielded by `LLMRouter.stream`):

    provider  {provider, model}            first committed token is about to arrive from this provider
    token     {text}
    restart   {reason, from, to, discarded_chars}   a provider failed BEFORE the commit point; whatever it
                                           produced was discarded and the next provider starts from scratch
    done      {provider, model, restarts}
    error     {code, message, partial, ...}   terminal

Commit point: tokens are held back until `commit_chars` characters have arrived (or the stream ends).
A failure before that is invisible to the user apart from a `restart` notice. A failure after it cannot be
hidden any more, so the router reports `error` with `partial: true` and the UI offers Retry.
Retrieval is never re-run: the same prompt is simply replayed against the next provider.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Sequence

from app.llm.base import ChatMessage, LLMProvider, ProviderError

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RouterEvent:
    type: str
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProviderState:
    cooldown_until: float = 0.0
    failures: int = 0
    successes: int = 0
    last_error: str | None = None
    last_status: int | None = None
    last_detail: str | None = None  # provider error text (HTTP status + start of the error body), never user text


class LLMRouter:
    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        commit_chars: int = 120,
        first_token_timeout: float = 20.0,
        stall_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.providers = list(providers)
        self.commit_chars = commit_chars
        self.first_token_timeout = first_token_timeout
        self.stall_timeout = stall_timeout
        self._clock = clock
        self._state = {p.label: ProviderState() for p in self.providers}

    # ----------------------------------------------------------------- admin
    def status(self) -> list[dict]:
        now = self._clock()
        return [
            {
                "provider": p.name,
                "model": p.model,
                "available": self._state[p.label].cooldown_until <= now,
                "cooldown_remaining_s": round(max(0.0, self._state[p.label].cooldown_until - now), 1),
                "failures": self._state[p.label].failures,
                "successes": self._state[p.label].successes,
                "last_error": self._state[p.label].last_error,
                "last_status": self._state[p.label].last_status,
                "last_detail": self._state[p.label].last_detail,
            }
            for p in self.providers
        ]

    def reset_cooldowns(self) -> None:
        for st in self._state.values():
            st.cooldown_until = 0.0

    # ---------------------------------------------------------------- stream
    async def stream(
        self, messages: list[ChatMessage], *, temperature: float = 0.2, max_tokens: int = 900
    ) -> AsyncIterator[RouterEvent]:
        now = self._clock()
        attempts: list[dict] = []
        plan: list[LLMProvider] = []
        for p in self.providers:
            st = self._state[p.label]
            if st.cooldown_until > now:
                attempts.append(
                    {
                        "provider": p.label, "kind": "cooldown", "was": st.last_error, "status": st.last_status,
                        "remaining_s": round(st.cooldown_until - now, 1),
                    }
                )
            else:
                plan.append(p)

        restarts = 0
        for i, provider in enumerate(plan):
            if i > 0:
                restarts += 1
                yield RouterEvent(
                    "restart",
                    {
                        "reason": attempts[-1]["kind"],
                        "from": plan[i - 1].label,
                        "to": provider.label,
                        "discarded_chars": attempts[-1].get("discarded_chars", 0),
                    },
                )

            committed = False
            buffer: list[str] = []
            buffered_chars = 0
            produced = 0
            it = provider.stream(messages, temperature=temperature, max_tokens=max_tokens).__aiter__()
            try:
                while True:
                    timeout = self.first_token_timeout if produced == 0 else self.stall_timeout
                    try:
                        token = await asyncio.wait_for(it.__anext__(), timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise ProviderError(
                            f"{provider.label}: no data for {timeout:.0f}s", provider=provider.name, kind="timeout"
                        ) from None
                    if not token:
                        continue
                    produced += 1
                    if committed:
                        yield RouterEvent("token", {"text": token})
                        continue
                    buffer.append(token)
                    buffered_chars += len(token)
                    if buffered_chars >= self.commit_chars:
                        committed = True
                        yield RouterEvent("provider", {"provider": provider.name, "model": provider.model})
                        for t in buffer:
                            yield RouterEvent("token", {"text": t})
                        buffer.clear()

                if produced == 0:
                    raise ProviderError(f"{provider.label}: empty response", provider=provider.name, kind="empty")
                if not committed:  # short answer: the stream ended before the commit threshold
                    yield RouterEvent("provider", {"provider": provider.name, "model": provider.model})
                    for t in buffer:
                        yield RouterEvent("token", {"text": t})
                st = self._state[provider.label]
                st.successes += 1
                st.last_error = st.last_status = st.last_detail = None
                yield RouterEvent("done", {"provider": provider.name, "model": provider.model, "restarts": restarts})
                return

            except ProviderError as exc:
                err = exc
            except Exception as exc:  # noqa: BLE001 - a provider bug must not take the request down
                log.exception("unexpected provider failure", extra={"provider": provider.label})
                err = ProviderError(f"{provider.label}: {type(exc).__name__}", provider=provider.name, kind="internal")
            finally:
                await _aclose(it)

            # ---- failure path (only reached via the except clauses above)
            st = self._state[provider.label]
            st.failures += 1
            st.last_error = err.kind
            st.last_status = err.status
            st.last_detail = str(err)[:300]
            st.cooldown_until = self._clock() + err.cooldown_seconds
            log.warning(
                "provider failed",
                extra={
                    "provider": provider.label, "kind": err.kind, "status": err.status,
                    "committed": committed, "cooldown_s": round(err.cooldown_seconds, 1),
                    "detail": st.last_detail,
                },
            )
            if committed:
                yield RouterEvent(
                    "error",
                    {
                        "code": "stream_interrupted",
                        "message": "The response was interrupted. Retry to generate it again.",
                        "partial": True,
                        "provider": provider.label,
                        "kind": err.kind,
                    },
                )
                return
            attempts.append(
                {"provider": provider.label, "kind": err.kind, "status": err.status, "discarded_chars": buffered_chars}
            )

        waits = [self._state[p.label].cooldown_until - self._clock() for p in self.providers]
        retry_after = round(max(0.0, min(waits)), 1) if waits else None
        yield RouterEvent(
            "error",
            {
                "code": "all_providers_failed",
                "message": _exhausted_message(attempts, retry_after),
                "partial": False,
                "attempts": [
                    {k: a[k] for k in ("provider", "kind", "status", "was", "remaining_s") if a.get(k) is not None}
                    for a in attempts
                ],
                "retry_after_s": retry_after,
            },
        )


_KIND_TEXT = {
    "rate_limit": "rate-limited",
    "unavailable": "unavailable",
    "timeout": "timed out",
    "connection": "not reachable",
    "auth": "rejected the API key",
    "not_found": "model not found",
    "bad_request": "rejected the request (too large?)",
    "blocked": "blocked the content",
    "empty": "returned nothing",
    "protocol": "sent a malformed reply",
    "internal": "internal error",
    "error": "failed",
}


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 90:
        return f"{round(seconds / 60)} min"
    return f"{max(1, round(seconds))} s"


def _exhausted_message(attempts: list[dict], retry_after: float | None) -> str:
    """Say *why* nothing could answer. 'Everything is rate-limited' was hiding dead model ids and bad keys."""
    if not attempts:
        return "No language model is configured. Check LLM_CHAIN and the API keys."
    parts = []
    for a in attempts:
        if a["kind"] == "cooldown":
            why = _KIND_TEXT.get(a.get("was") or "", "failed recently")
            parts.append(f"{a['provider']} {why}, skipped for another {_duration(a['remaining_s'])}")
        else:
            parts.append(f"{a['provider']} {_KIND_TEXT.get(a['kind'], 'failed')}")
    msg = "No language model could answer: " + "; ".join(parts) + "."
    if retry_after:
        msg += f" Earliest retry in {_duration(retry_after)}."
    return msg


async def _aclose(it) -> None:
    close = getattr(it, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:  # noqa: BLE001
            pass
