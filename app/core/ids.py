"""Stable identifiers shared by ingestion and retrieval."""
from __future__ import annotations

import uuid

_NAMESPACE = uuid.UUID("6f1d2c3e-4b5a-4c6d-8e7f-0a1b2c3d4e5f")


def chunk_point_id(document_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id, so re-ingesting the same document overwrites in place."""
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk_index}"))
