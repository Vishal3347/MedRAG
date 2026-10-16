"""Thin HTTP streaming abstraction so providers can be tested with mocked callers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from app.llm.base import ProviderError


class UpstreamHTTPError(Exception):
    def __init__(self, status: int, body: str = "", retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


class UpstreamTimeout(Exception):
    pass


class UpstreamConnectionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class StreamTimeouts:
    connect: float = 10.0
    read: float = 30.0


class HTTPStreamer(Protocol):
    def stream_lines(
        self, url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeouts: StreamTimeouts
    ) -> AsyncIterator[str]: ...


def _retry_after_header(headers: Any) -> float | None:
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


class HttpxStreamer:
    """Production streamer built on httpx. The AsyncClient is created lazily and reused."""

    def __init__(self, client: Any | None = None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=10))
        return self._client

    async def stream_lines(
        self, url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeouts: StreamTimeouts
    ) -> AsyncIterator[str]:
        import httpx

        client = self._get_client()
        timeout = httpx.Timeout(connect=timeouts.connect, read=timeouts.read, write=10.0, pool=10.0)
        try:
            async with client.stream("POST", url, headers=headers, json=json_body, timeout=timeout) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:2000]
                    raise UpstreamHTTPError(resp.status_code, body, _retry_after_header(resp.headers))
                async for line in resp.aiter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise UpstreamTimeout(str(exc)) from exc
        except httpx.TransportError as exc:
            raise UpstreamConnectionError(str(exc)) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def parse_retry_delay(body: str) -> float | None:
    """Gemini reports the wait in the JSON error body, e.g. "retryDelay": "32s"."""
    m = _RETRY_DELAY_RE.search(body or "")
    return float(m.group(1)) if m else None


def map_upstream_error(provider: str, exc: Exception) -> ProviderError:
    """Translate transport-level failures into a ProviderError with a retry classification."""
    if isinstance(exc, UpstreamTimeout):
        return ProviderError(f"{provider}: timeout", provider=provider, kind="timeout")
    if isinstance(exc, UpstreamConnectionError):
        return ProviderError(f"{provider}: connection failed", provider=provider, kind="connection")
    if isinstance(exc, UpstreamHTTPError):
        status = exc.status
        retry_after = exc.retry_after if exc.retry_after is not None else parse_retry_delay(exc.body)
        low = exc.body.lower()
        if status == 429:
            if retry_after is None and ("per day" in low or "perday" in low or "daily" in low):
                retry_after = 900.0  # daily quota exhausted: do not hammer it every minute
            kind = "rate_limit"
        elif status in (500, 502, 503, 504, 529):
            kind = "unavailable"
        elif status == 408:
            kind = "timeout"
        elif status in (401, 403):
            kind = "auth"
        elif status == 404:
            kind = "not_found"
        elif status in (400, 413, 422):
            kind = "bad_request"
        else:
            kind = "error"
        return ProviderError(
            f"{provider}: HTTP {status} {exc.body[:200]!r}", provider=provider, kind=kind, status=status,
            retry_after=retry_after,
        )
    return ProviderError(f"{provider}: {type(exc).__name__}: {exc}", provider=provider, kind="internal")
