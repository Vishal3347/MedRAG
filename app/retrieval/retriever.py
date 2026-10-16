"""Staged hybrid retrieval: dense + sparse (top-N each) -> RRF -> dedupe -> top-K -> evidence gate."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.retrieval.fusion import dedupe, rrf_fuse
from app.retrieval.models import Candidate, Hit, RetrievalResult
from app.retrieval.sparse import encode_query

log = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(
        self,
        store: Any,
        embedder: Any,
        *,
        candidates: int = 20,
        top_k: int = 5,
        rrf_k: int = 60,
        min_dense_score: float = 0.45,
    ):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.min_dense_score = min_dense_score

    async def retrieve(self, query: str, *, document_id: str | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        try:
            dense_vec = await asyncio.to_thread(self.embedder.embed_query, query)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"query embedding failed: {exc}") from exc

        indices, values = encode_query(query)
        dense_task = asyncio.to_thread(self.store.dense_search, dense_vec, self.candidates, document_id)
        if indices:
            sparse_task = asyncio.to_thread(self.store.sparse_search, indices, values, self.candidates, document_id)
            dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)
        else:
            dense_res, sparse_res = await dense_task, []
        if isinstance(dense_res, BaseException):
            raise RetrievalError(f"dense search failed: {dense_res}") from dense_res
        if isinstance(sparse_res, BaseException):
            log.warning("sparse search failed; continuing dense-only", extra={"error": str(sparse_res)})
            sparse_res = []

        result = self._fuse(dense_res, sparse_res)
        result.stats["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        return result

    def _fuse(self, dense: list[Hit], sparse: list[Hit]) -> RetrievalResult:
        by_id: dict[str, Candidate] = {}
        for hit in dense:
            by_id[hit.id] = Candidate(hit.id, hit.payload, 0.0, dense_score=hit.score)
        for hit in sparse:
            cand = by_id.get(hit.id)
            if cand is None:
                by_id[hit.id] = Candidate(hit.id, hit.payload, 0.0, sparse_score=hit.score)
            else:
                cand.sparse_score = hit.score

        fused = rrf_fuse([[h.id for h in dense], [h.id for h in sparse]], k=self.rrf_k)
        ranked: list[Candidate] = []
        for doc_id, score in fused:
            cand = by_id[doc_id]
            cand.rrf_score = score
            ranked.append(cand)

        unique = dedupe(ranked, text_of=lambda c: c.text, hash_of=lambda c: c.payload.get("content_hash"))
        top = unique[: self.top_k]

        best_dense = max((h.score for h in dense), default=None)
        stats = {"dense_candidates": len(dense), "sparse_candidates": len(sparse), "fused": len(fused)}
        if not top:
            return RetrievalResult([], False, best_dense, "no_results", stats)
        if best_dense is None or best_dense < self.min_dense_score:
            return RetrievalResult(top, False, best_dense, "low_similarity", stats)
        return RetrievalResult(top, True, best_dense, "ok", stats)
