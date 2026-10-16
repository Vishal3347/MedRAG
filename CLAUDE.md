# MedRAG: working notes for Claude Code

Medical-textbook RAG chatbot: FastAPI backend (`app/`), Next.js frontend (`frontend/`), Qdrant Cloud, Groq -> Gemini -> Ollama failover.

## Commands
- Backend deps: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt`
- Tests: `pytest` (router, chunking, retrieval and API tests run offline with fakes)
- Inspect chunking before spending embedding time: `python -m app.scripts.ingest_pdf --pdf-path book.pdf --dry-run --pages 1-60 --dump-chunks data/sample.jsonl`
- Ingest: `python -m app.scripts.ingest_pdf --pdf-path book.pdf`
- Run API: `uvicorn app.api.main:app --reload`
- Frontend: `cd frontend && npm install && npm run dev` (`npm run typecheck`)

## Invariants worth keeping
- Retrieval runs once per request; failover replays the same prompt (see `app/llm/router.py`).
- Tokens are buffered until `STREAM_COMMIT_CHARS`; failures before that emit `restart`, after that `error` with `partial: true`.
- If best dense score < `MIN_DENSE_SCORE` the LLM is not called at all.
- Point ids are `uuid5(document_id:chunk_index)`; re-ingesting is idempotent and skips unchanged chunks.
- Never log question or answer text.
- `app/llm/*`, `app/retrieval/{sparse,fusion,query,retriever}.py` and `app/ingestion/*` must stay importable without qdrant/fastapi/fitz/torch (tests rely on it).
