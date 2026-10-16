"""python -m app.scripts.ingest_pdf --pdf-path book.pdf

Idempotent: safe to re-run after an interruption; unchanged chunks are skipped.
Tip: start with `--dry-run --pages 1-60 --dump-chunks data/sample.jsonl` and read the chunks before the full run.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ingestion.pipeline import IngestOptions, run_ingest


def _page_range(value: str) -> tuple[int, int | None]:
    start, _, end = value.partition("-")
    return int(start), (int(end) if end else int(start))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Index a medical PDF into Qdrant (dense + sparse).")
    p.add_argument("--pdf-path", required=True, type=Path)
    p.add_argument("--title", help="Document title (default: PDF metadata, else file name)")
    p.add_argument("--document-id", help="Stable id (default: hash of the PDF bytes)")
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--overlap-tokens", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--pages", type=_page_range, metavar="START-END", help="Only process this 1-based page range")
    p.add_argument("--no-ocr", action="store_true", help="Disable the Tesseract fallback for scanned pages")
    p.add_argument("--ocr-lang", default="eng")
    p.add_argument("--dump-chunks", type=Path, help="Write the chunks to this JSONL file for inspection")
    p.add_argument("--dry-run", action="store_true", help="Extract and chunk only; no embeddings, no Qdrant")
    p.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection first")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    if not args.pdf_path.is_file():
        print(f"error: {args.pdf_path} does not exist", file=sys.stderr)
        return 2

    first, last = args.pages or (1, None)
    opts = IngestOptions(
        pdf_path=args.pdf_path,
        title=args.title,
        document_id=args.document_id,
        chunk_tokens=args.chunk_tokens or settings.chunk_tokens,
        overlap_tokens=args.overlap_tokens if args.overlap_tokens is not None else settings.chunk_overlap_tokens,
        batch_size=args.batch_size or settings.embedding_batch_size,
        ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
        first_page=first,
        last_page=last,
        dump_chunks=args.dump_chunks,
        dry_run=args.dry_run,
        recreate=args.recreate,
    )
    report = run_ingest(opts, settings)
    print(json.dumps(asdict(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
