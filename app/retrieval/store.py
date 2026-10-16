"""Qdrant Cloud access: collection setup, idempotent upserts, dense + sparse search."""
from __future__ import annotations

import logging
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.retrieval.models import Hit, PointRecord

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim

    @classmethod
    def from_settings(cls, settings, dim: int | None = None) -> "QdrantStore":
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL is not set. Create a free cluster at https://cloud.qdrant.io and fill .env")
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() or None,
            timeout=30,
        )
        return cls(client, settings.qdrant_collection, dim or settings.embedding_dim)

    # ---------------------------------------------------------------- collection
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            log.warning("deleting collection", extra={"collection": self.collection})
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors[DENSE].size if isinstance(vectors, dict) else getattr(vectors, "size", None)
            if size is not None and size != self.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has {size}-dim vectors but the embedder produces {self.dim}. "
                    "Re-run ingestion with --recreate (or use another QDRANT_COLLECTION)."
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        for field, schema in (
            ("document_id", qm.PayloadSchemaType.KEYWORD),
            ("chunk_index", qm.PayloadSchemaType.INTEGER),
            ("page_start", qm.PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
        log.info("created collection", extra={"collection": self.collection, "dim": self.dim})

    # -------------------------------------------------------------------- writes
    def existing_payloads(self, ids: Sequence[str], fields: Sequence[str], batch: int = 256) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for i in range(0, len(ids), batch):
            for rec in self.client.retrieve(
                self.collection, ids=list(ids[i : i + batch]), with_payload=list(fields), with_vectors=False
            ):
                found[str(rec.id)] = dict(rec.payload or {})
        return found

    def upsert(self, records: Sequence[PointRecord]) -> None:
        points = [
            qm.PointStruct(
                id=r.id,
                vector={DENSE: r.dense, SPARSE: qm.SparseVector(indices=r.sparse_indices, values=r.sparse_values)},
                payload=r.payload,
            )
            for r in records
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def _doc_filter(self, document_id: str | None) -> qm.Filter | None:
        if not document_id:
            return None
        return qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])

    def delete_stale(self, document_id: str, keep_count: int) -> None:
        """Remove points of this document whose chunk_index is beyond the freshly ingested range."""
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
                qm.FieldCondition(key="chunk_index", range=qm.Range(gte=keep_count)),
            ]
        )
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=flt), wait=True)

    def count(self, document_id: str | None = None) -> int:
        return int(self.client.count(self.collection, count_filter=self._doc_filter(document_id), exact=True).count)

    # --------------------------------------------------------------------- reads
    def dense_search(self, vector: list[float], limit: int, document_id: str | None = None) -> list[Hit]:
        res = self.client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            query_filter=self._doc_filter(document_id),
            with_payload=True,
            with_vectors=False,
        )
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def sparse_search(
        self, indices: list[int], values: list[float], limit: int, document_id: str | None = None
    ) -> list[Hit]:
        res = self.client.query_points(
            self.collection,
            query=qm.SparseVector(indices=indices, values=values),
            using=SPARSE,
            limit=limit,
            query_filter=self._doc_filter(document_id),
            with_payload=True,
            with_vectors=False,
        )
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def get_chunk(self, chunk_id: str) -> dict | None:
        recs = self.client.retrieve(self.collection, ids=[chunk_id], with_payload=True, with_vectors=False)
        if not recs:
            return None
        return {"id": str(recs[0].id), **(recs[0].payload or {})}

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        offset = None
        fields = ["document_id", "document_title", "page_start", "page_end", "chapter"]
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=500, offset=offset, with_payload=fields, with_vectors=False
            )
            for p in points:
                pl = p.payload or {}
                doc_id = pl.get("document_id")
                if not doc_id:
                    continue
                d = docs.setdefault(
                    doc_id,
                    {
                        "document_id": doc_id,
                        "title": pl.get("document_title", doc_id),
                        "chunks": 0,
                        "page_start": pl.get("page_start", 0),
                        "page_end": pl.get("page_end", 0),
                        "_chapters": set(),
                    },
                )
                d["chunks"] += 1
                d["page_start"] = min(d["page_start"], pl.get("page_start", d["page_start"]))
                d["page_end"] = max(d["page_end"], pl.get("page_end", d["page_end"]))
                if pl.get("chapter"):
                    d["_chapters"].add(pl["chapter"])
            if offset is None:
                break
        out = []
        for d in docs.values():
            d["chapters"] = len(d.pop("_chapters"))
            out.append(d)
        return sorted(out, key=lambda d: d["title"])

    def points_count(self) -> int:
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    def ping(self) -> bool:
        return bool(self.client.collection_exists(self.collection))
