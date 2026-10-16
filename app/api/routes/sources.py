from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/v1")


@router.get("/sources/{chunk_id}")
async def get_source(chunk_id: str, request: Request) -> dict:
    try:
        uuid.UUID(chunk_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown source.") from None
    store = request.app.state.container.store
    try:
        chunk = await asyncio.to_thread(store.get_chunk, chunk_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Source lookup is unavailable.") from exc
    if chunk is None:
        raise HTTPException(status_code=404, detail="Unknown source.")
    return chunk
