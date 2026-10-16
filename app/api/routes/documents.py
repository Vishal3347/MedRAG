from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import cached_documents

router = APIRouter(prefix="/api/v1")


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    try:
        docs = await cached_documents(request.app.state.container)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Document index is unavailable.") from exc
    return {"documents": docs}
