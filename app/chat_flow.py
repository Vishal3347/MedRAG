"""Transport-independent chat pipeline: retrieve -> evidence gate -> prompt -> router stream.

Yields (event, data) pairs. The FastAPI route wraps them as SSE; the Streamlit app consumes them directly.
Must stay importable without fastapi/streamlit.
"""
from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Sequence

from app.llm.prompts import INSUFFICIENT_EVIDENCE, build_messages, estimate_tokens
from app.retrieval.query import build_retrieval_query

log = logging.getLogger(__name__)

_HEADING_MARK = re.compile(r"^#{1,6}\s+", re.M)


def _snippet(text: str, limit: int = 280) -> str:
    flat = " ".join(_HEADING_MARK.sub("", text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _source_out(ref: int, cand) -> dict:
    pl = cand.payload
    return {
        "ref": f"S{ref}",
        "chunk_id": cand.id,
        "document_id": pl.get("document_id", ""),
        "document_title": pl.get("document_title", ""),
        "chapter": pl.get("chapter", ""),
        "section_heading": pl.get("section_heading", ""),
        "page_start": pl.get("page_start", 0),
        "page_end": pl.get("page_end", 0),
        "snippet": _snippet(cand.text),
        "score": round(cand.dense_score, 4) if cand.dense_score is not None else None,
    }


async def chat_events(
    retriever: Any, router: Any, settings: Any, question: str, history: Sequence[tuple[str, str]],
    document_id: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    try:
        query = build_retrieval_query(history, question)
        result = await retriever.retrieve(query, document_id=document_id)
    except Exception:  # noqa: BLE001
        log.exception("retrieval failed")
        yield "error", {"code": "retrieval_failed", "message": "The textbook index is unavailable right now.", "partial": False}
        return

    sources = [_source_out(i, cand) for i, cand in enumerate(result.chunks, start=1)]
    log.info(
        "retrieved",
        extra={"confident": result.confident, "best_dense": result.best_dense, "chunks": len(sources), **result.stats},
    )
    yield "sources", {"sources": sources, "confident": result.confident, "best_score": result.best_dense, "reason": result.reason}

    if not result.confident:
        yield "token", {"text": INSUFFICIENT_EVIDENCE}
        yield "done", {"provider": None, "model": None, "restarts": 0, "reason": "insufficient_evidence"}
        return

    messages = build_messages(
        question, result.chunks, history,
        max_history=settings.max_history_messages, max_input_tokens=settings.llm_max_input_tokens,
    )
    log.info("prompt built", extra={"est_input_tokens": sum(estimate_tokens(m.content) for m in messages)})
    async for ev in router.stream(messages, temperature=settings.llm_temperature, max_tokens=settings.llm_max_output_tokens):
        yield ev.type, ev.data
