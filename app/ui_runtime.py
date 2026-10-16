"""Synchronous bridge for UIs that are not async (Streamlit).

One long-lived event loop runs in a background thread, so the shared httpx client, router cooldown state,
embedder and Qdrant connection are created once and reused across reruns and sessions.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Iterator, Sequence

from app.chat_flow import chat_events


class Runtime:
    def __init__(self, retriever: Any, router: Any, settings: Any, embedder: Any = None, streamer: Any = None):
        self.retriever, self.router, self.settings = retriever, router, settings
        self.embedder, self.streamer = embedder, streamer
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="medrag-loop", daemon=True).start()

    def stream(self, question: str, history: Sequence[tuple[str, str]]) -> Iterator[tuple[str, dict]]:
        q: queue.Queue = queue.Queue()

        async def pump() -> None:
            try:
                async for item in chat_events(self.retriever, self.router, self.settings, question, history):
                    q.put(item)
            except Exception as exc:  # noqa: BLE001
                q.put(("error", {"code": "internal", "message": f"Unexpected error: {type(exc).__name__}", "partial": False}))
            finally:
                q.put(None)

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        while (item := q.get()) is not None:
            yield item

    def provider_status(self) -> list[dict]:
        return self.router.status()


def build_runtime(settings: Any) -> Runtime:
    from app.core.embeddings import make_embedder
    from app.llm.factory import build_router
    from app.retrieval.retriever import HybridRetriever
    from app.retrieval.store import QdrantStore

    embedder = make_embedder(settings)
    store = QdrantStore.from_settings(settings)
    retriever = HybridRetriever(
        store, embedder, candidates=settings.candidates_per_retriever, top_k=settings.context_chunks,
        rrf_k=settings.rrf_k, min_dense_score=settings.min_dense_score,
    )
    router, streamer = build_router(settings)
    embedder.warmup()
    return Runtime(retriever, router, settings, embedder, streamer)
