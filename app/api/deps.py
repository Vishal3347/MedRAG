from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request

from app.core.ratelimit import SlidingWindowLimiter

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Any
    retriever: Any
    router: Any
    store: Any
    embedder: Any = None
    limiter: SlidingWindowLimiter | None = None
    streamer: Any = None
    docs_cache: tuple[float, list[dict]] | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        close = getattr(self.streamer, "aclose", None)
        if close is not None:
            await close()


async def build_container(settings: Any) -> Container:
    from app.core.embeddings import make_embedder
    from app.llm.factory import build_router
    from app.retrieval.retriever import HybridRetriever
    from app.retrieval.store import QdrantStore

    embedder = make_embedder(settings)
    store = QdrantStore.from_settings(settings)
    retriever = HybridRetriever(
        store,
        embedder,
        candidates=settings.candidates_per_retriever,
        top_k=settings.context_chunks,
        rrf_k=settings.rrf_k,
        min_dense_score=settings.min_dense_score,
    )
    router, streamer = build_router(settings)
    try:  # load the model now so the first question is not slow; never block startup on failure
        await asyncio.to_thread(embedder.warmup)
    except Exception:  # noqa: BLE001
        log.exception("embedding model warmup failed")
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    return Container(settings, retriever, router, store, embedder, limiter, streamer)


def get_container(request: Request) -> Container:
    return request.app.state.container


def client_ip(request: Request) -> str:
    c = request.app.state.container
    if getattr(c.settings, "trust_proxy", False):
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    limiter = request.app.state.container.limiter
    if limiter is None:
        return
    allowed, retry_after = limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many questions in a short time. Please wait a moment.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


async def cached_documents(c: Container, ttl_s: float = 60.0) -> list[dict]:
    now = time.monotonic()
    if c.docs_cache and now - c.docs_cache[0] < ttl_s:
        return c.docs_cache[1]
    docs = await asyncio.to_thread(c.store.list_documents)
    c.docs_cache = (now, docs)
    return docs
