from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request

from app.core.security import require_admin

router = APIRouter(prefix="/api/v1/admin", dependencies=[Depends(require_admin)])


@router.get("/status")
async def status(request: Request) -> dict:
    c = request.app.state.container
    try:
        points = await asyncio.to_thread(c.store.points_count)
    except Exception:  # noqa: BLE001
        points = None
    return {
        "providers": c.router.status(),
        "indexed_chunks": points,
        "retrieval": {
            "candidates_per_retriever": c.retriever.candidates,
            "top_k": c.retriever.top_k,
            "min_dense_score": c.retriever.min_dense_score,
        },
    }


@router.get("/retrieve")
async def retrieve_debug(request: Request, q: str = Query(min_length=1, max_length=500)) -> dict:
    """Inspect scores for a query. Use it to calibrate MIN_DENSE_SCORE: run a handful of questions the book
    answers and a handful it cannot, and pick a threshold between the two groups of best_dense values."""
    c = request.app.state.container
    result = await c.retriever.retrieve(q)
    return {
        "confident": result.confident,
        "best_dense": result.best_dense,
        "reason": result.reason,
        "stats": result.stats,
        "chunks": [
            {
                "id": k.id,
                "dense": k.dense_score,
                "sparse": k.sparse_score,
                "rrf": round(k.rrf_score, 5),
                "chapter": k.payload.get("chapter"),
                "section": k.payload.get("section_heading"),
                "pages": [k.payload.get("page_start"), k.payload.get("page_end")],
                "preview": k.text[:160],
            }
            for k in result.chunks
        ],
    }


@router.post("/providers/reset")
async def reset_providers(request: Request) -> dict:
    request.app.state.container.router.reset_cooldowns()
    return {"status": "ok", "providers": request.app.state.container.router.status()}
