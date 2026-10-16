"""End-to-end ingestion: PDF -> cleaned blocks -> chunks -> embeddings -> idempotent Qdrant upsert."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.ids import chunk_point_id
from app.ingestion.chunking import Chunker
from app.ingestion.cleaning import find_running_keys
from app.ingestion.models import Chunk
from app.ingestion.structure import build_blocks, compute_body_size
from app.ingestion.tokens import TokenCounter, approx_token_count

log = logging.getLogger(__name__)

META_FIELDS = ("content_hash", "document_title", "page_start", "page_end", "chapter", "section_heading")


@dataclass(slots=True)
class IngestOptions:
    pdf_path: Path
    title: str | None = None
    document_id: str | None = None
    chunk_tokens: int = 600
    overlap_tokens: int = 80
    batch_size: int = 32
    ocr: bool = True
    ocr_lang: str = "eng"
    first_page: int = 1
    last_page: int | None = None
    dump_chunks: Path | None = None
    dry_run: bool = False
    recreate: bool = False


@dataclass(slots=True)
class IngestReport:
    document_id: str = ""
    title: str = ""
    pages: int = 0
    pages_by_source: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    tokens_min: int = 0
    tokens_max: int = 0
    tokens_avg: int = 0
    embedded: int = 0
    skipped_unchanged: int = 0
    stored_count: int | None = None
    elapsed_s: float = 0.0


def compute_document_id(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return "doc_" + digest.hexdigest()[:16]


def prepare_chunks(opts: IngestOptions, count_tokens: TokenCounter = approx_token_count) -> tuple[list[Chunk], IngestReport]:
    from app.ingestion.extract import iter_raw_pages, page_count, read_pdf_title

    report = IngestReport()
    report.document_id = opts.document_id or compute_document_id(opts.pdf_path)
    report.title = opts.title or read_pdf_title(opts.pdf_path) or opts.pdf_path.stem.replace("_", " ").replace("-", " ")
    total = page_count(opts.pdf_path)
    pages = []
    for raw in iter_raw_pages(
        opts.pdf_path, ocr=opts.ocr, ocr_lang=opts.ocr_lang, first_page=opts.first_page, last_page=opts.last_page
    ):
        pages.append(raw)
        if raw.number % 50 == 0:
            log.info("extracted %d/%d pages", raw.number, total)
    report.pages = len(pages)
    report.pages_by_source = dict(Counter(p.source for p in pages))

    body = compute_body_size(pages)
    running = find_running_keys(pages, body)
    log.info("body font %.1fpt, %d running header/footer patterns removed", body, len(running))
    blocks = build_blocks(pages, body, running)
    log.info("%d structural blocks (%d headings)", len(blocks), sum(b.is_heading for b in blocks))

    chunker = Chunker(
        document_id=report.document_id,
        document_title=report.title,
        target_tokens=opts.chunk_tokens,
        overlap_tokens=opts.overlap_tokens,
        count_tokens=count_tokens,
    )
    chunks = chunker.chunk(blocks)
    if chunks:
        sizes = [c.token_count for c in chunks]
        report.chunks = len(chunks)
        report.tokens_min, report.tokens_max = min(sizes), max(sizes)
        report.tokens_avg = round(sum(sizes) / len(sizes))
    return chunks, report


def dump_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({"id": chunk_point_id(c.document_id, c.chunk_index), **c.payload()}, ensure_ascii=False) + "\n")


def _retry(fn: Callable[[], Any], attempts: int = 4, base_delay: float = 1.5) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("upsert failed (%s); retry %d/%d in %.1fs", exc, attempt, attempts - 1, delay)
            time.sleep(delay)


def run_ingest(opts: IngestOptions, settings: Any) -> IngestReport:
    from app.core.embeddings import Embedder
    from app.retrieval.models import PointRecord
    from app.retrieval.sparse import encode_document
    from app.retrieval.store import QdrantStore

    t0 = time.perf_counter()
    if opts.dry_run:
        chunks, report = prepare_chunks(opts)
        if opts.dump_chunks:
            dump_jsonl(chunks, opts.dump_chunks)
        report.elapsed_s = round(time.perf_counter() - t0, 1)
        return report

    embedder = Embedder(settings.embedding_model, dim=settings.embedding_dim, batch_size=opts.batch_size)
    chunks, report = prepare_chunks(opts, embedder.token_counter())
    if opts.dump_chunks:
        dump_jsonl(chunks, opts.dump_chunks)
    if not chunks:
        raise RuntimeError("No chunks were produced. Is the PDF scanned? Install OCR support (requirements-ocr.txt).")
    if opts.chunk_tokens > embedder.max_tokens:
        log.warning(
            "chunk size (%d tokens) exceeds the embedding model's limit (%d): the dense vector only sees the start "
            "of each chunk (BM25 and the LLM still see all of it). Use --chunk-tokens %d to avoid this.",
            opts.chunk_tokens, embedder.max_tokens, embedder.max_tokens - 60,
        )

    store = QdrantStore.from_settings(settings, dim=embedder.dimension)
    store.ensure_collection(recreate=opts.recreate)

    ids = [chunk_point_id(c.document_id, c.chunk_index) for c in chunks]
    existing = store.existing_payloads(ids, META_FIELDS)
    todo: list[tuple[str, Chunk]] = []
    for pid, c in zip(ids, chunks):
        meta = {
            "content_hash": c.content_hash, "document_title": c.document_title, "page_start": c.page_start,
            "page_end": c.page_end, "chapter": c.chapter, "section_heading": c.section_heading,
        }
        if existing.get(pid) == meta:
            report.skipped_unchanged += 1
        else:
            todo.append((pid, c))
    log.info("%d chunks to embed, %d unchanged and skipped", len(todo), report.skipped_unchanged)

    for i in range(0, len(todo), opts.batch_size):
        batch = todo[i : i + opts.batch_size]
        texts = [c.embed_text for _, c in batch]
        vectors = embedder.embed_documents(texts)
        records = []
        for (pid, c), vec, text in zip(batch, vectors, texts):
            idx, val = encode_document(text)
            records.append(PointRecord(pid, vec, idx, val, c.payload()))
        _retry(lambda r=records: store.upsert(r))
        report.embedded += len(batch)
        log.info("upserted %d/%d", report.embedded, len(todo))

    store.delete_stale(report.document_id, keep_count=len(chunks))
    report.stored_count = store.count(report.document_id)
    if report.stored_count != len(chunks):
        log.error("expected %d points but Qdrant reports %d", len(chunks), report.stored_count)
    report.elapsed_s = round(time.perf_counter() - t0, 1)
    return report
