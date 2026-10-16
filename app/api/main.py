from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import Container, build_container
from app.api.routes import admin, chat, documents, health, sources
from app.core.config import get_settings
from app.core.logging import configure_logging, request_id_var

log = logging.getLogger("app.access")
_VALID_RID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


class RequestContextMiddleware:
    """Pure-ASGI middleware (safe for streaming responses): request id + one structured access-log line.
    Request bodies, questions and answers are never logged."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _VALID_RID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        started = time.perf_counter()
        state = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "route": scope["path"],
                    "status": state["status"],
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
            request_id_var.reset(token)


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container is None:
            app.state.container = await build_container(settings)
        yield
        await app.state.container.aclose()

    app = FastAPI(title="MedRAG", version="1.0.0", lifespan=lifespan, redoc_url=None)
    if container is not None:
        app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Admin-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(sources.router)
    app.include_router(admin.router)
    return app


app = create_app()
