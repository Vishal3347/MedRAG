"""Admin endpoint protection."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request


def require_admin(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
    configured = request.app.state.container.settings.admin_api_key.get_secret_value()
    if not configured:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled (ADMIN_API_KEY is not set).")
    if not x_admin_key or not hmac.compare_digest(x_admin_key.encode(), configured.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header.")
