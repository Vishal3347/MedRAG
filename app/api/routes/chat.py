from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import rate_limit
from app.api.schemas import ChatRequest
from app.api.sse import sse
from app.chat_flow import chat_events
from app.core.logging import request_id_var

router = APIRouter(prefix="/api/v1")


@router.post("/chat/stream", dependencies=[Depends(rate_limit)])
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    c = request.app.state.container
    rid = request_id_var.get()
    history = [(m.role, m.content) for m in body.history]

    async def events() -> AsyncIterator[str]:
        yield sse("meta", {"request_id": rid})
        async for event, data in chat_events(c.retriever, c.router, c.settings, body.question, history, body.document_id):
            yield sse(event, data)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
