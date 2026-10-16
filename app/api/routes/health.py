from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    c = request.app.state.container
    checks: dict[str, object] = {}
    ok = True
    try:
        collection_ok = await asyncio.to_thread(c.store.ping)
        checks["qdrant"] = bool(collection_ok)
        if collection_ok:
            count = await asyncio.to_thread(c.store.points_count)
            checks["indexed_chunks"] = count
            ok = ok and count > 0
        else:
            ok = False
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = False
        checks["qdrant_error"] = type(exc).__name__
        ok = False
    embedder = getattr(c, "embedder", None)
    checks["embedder_loaded"] = True if embedder is None else bool(embedder.is_loaded)
    ok = ok and checks["embedder_loaded"]
    checks["llm_providers"] = len(c.router.providers)
    ok = ok and checks["llm_providers"] > 0
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)
