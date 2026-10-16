# MedRAG: a zero-cost RAG chatbot for a medical textbook

Ask questions about a ~1,000-page medical PDF and get answers grounded in the book, with **chapter + PDF page
citations** you can click to read the exact passage. Everything runs on free tiers.

```
PDF --PyMuPDF/pdfplumber/OCR--> clean blocks --structure-aware chunker (600 tok / 80 overlap)--+
                                                                                               v
                          local BGE embeddings + BM25 sparse vectors  --->  Qdrant Cloud (free)
                                                                                               |
question --> dense top-20 + sparse top-20 --> RRF --> dedupe --> top-5 --> evidence gate -------+
                                                                              |
                          confident? no -> "not enough support" (no LLM call, nearest passages shown)
                                    yes -> Groq -> Gemini -> Ollama (failover, same prompt, no re-search)
                                                                              |
                          SSE: meta, sources, provider, token, restart, done | error  --> Next.js UI
```

## Quick start

Prerequisites: Python 3.11+, Node 20+, a free [Qdrant Cloud](https://cloud.qdrant.io) cluster, and at least one of a
free [Groq](https://console.groq.com) or [Gemini](https://aistudio.google.com) API key.

```bash
cp .env.example .env            # fill QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY and/or GEMINI_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # small CPU build
pip install -r requirements-dev.txt
pytest                                                                  # offline, no keys needed

# 1. Look at the chunks before paying for embeddings (60 pages, no network):
python -m app.scripts.ingest_pdf --pdf-path book.pdf --dry-run --pages 1-60 --dump-chunks data/sample.jsonl
# 2. Full ingestion (idempotent; safe to re-run or resume):
python -m app.scripts.ingest_pdf --pdf-path book.pdf

uvicorn app.api.main:app --reload                                       # http://localhost:8000/docs
cd frontend && cp .env.local.example .env.local && npm install && npm run dev   # http://localhost:3000
```

No API keys yet? `LLM_CHAIN=mock:echo` runs the whole stack with a stub model that quotes the first retrieved
passage, which is handy for checking citations.

Scanned PDF? `pip install -r requirements-ocr.txt` and install the `tesseract` binary; pages with no text layer are
OCR'd automatically (`--no-ocr` to disable).

## Tune these three things on your own book

1. **Chunk size.** `bge-small-en-v1.5` embeds at most 512 tokens, so the spec's 600-token chunks are only partly
   seen by the dense vector (BM25 and the LLM still see everything). Ingestion warns about this. For best dense
   recall use `--chunk-tokens 450`. Re-ingest with `--recreate`.
2. **Evidence gate `MIN_DENSE_SCORE`.** Below this best-cosine value the bot refuses to answer instead of
   guessing. Calibrate: with `ADMIN_API_KEY` set, call
   `GET /api/v1/admin/retrieve?q=<question>` (header `X-Admin-Key`) for ~10 questions the book answers and ~10 it
   cannot (and for a few phrased differently from the book), then set the threshold between the two groups of
   `best_dense` values. The default 0.45 is a starting guess.
3. **Provider chain.** `LLM_CHAIN` is an ordered list of `provider:model`. See `docs/free_tier_audit.md`: the 70B
   model's free daily token cap is only about 20 questions a day, so a second Groq model and Gemini matter.

## How failover works

* One retrieval per request. If a provider fails, the **same prompt** goes to the next one.
* Tokens are held back until `STREAM_COMMIT_CHARS` (default 120) have arrived. A 429/503/timeout/stall/empty
  reply before that point is invisible except for a `restart` event (the UI shows "Switched to ... after a rate
  limit"). After that point the answer cannot be silently replaced, so the UI keeps the partial text and offers
  Retry (`error` event with `partial: true`).
* A failed provider sits out for `Retry-After` (or a default per failure type), so later requests skip it.
* `GET /api/v1/admin/status` shows provider state; `POST /api/v1/admin/providers/reset` clears cooldowns.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /ready` | Liveness; readiness (Qdrant reachable and indexed, model loaded, providers configured) |
| `GET /api/v1/documents` | Indexed documents with page range and chapter/chunk counts |
| `POST /api/v1/chat/stream` | SSE chat: `{question, history[], document_id?}` |
| `GET /api/v1/sources/{chunk_id}` | Exact stored chunk text and metadata (source inspector) |
| `GET/POST /api/v1/admin/*` | `X-Admin-Key` required; disabled when `ADMIN_API_KEY` is empty |

Every response carries `X-Request-ID`; logs are JSON with the request id. Questions and answers are never logged.
Chat is rate limited per IP (`RATE_LIMIT_PER_MINUTE`) to protect your free quotas.

## Deploy

`docker compose up --build` runs backend (:8000) and frontend (:3000). `--profile local-qdrant` / `--profile
local-llm` add local Qdrant / Ollama. The Docker image bakes in the embedding model and uses CPU torch.
See the hosting notes in `docs/free_tier_audit.md` (the backend needs ~1 GB RAM).

## Layout

```
app/core        config, logging + request ids, admin auth, rate limiter, embeddings, ids
app/ingestion   extract (PyMuPDF/pdfplumber/OCR), cleaning, structure, chunking, pipeline
app/retrieval   sparse (BM25-style), fusion (RRF + dedupe), store (Qdrant), retriever, query
app/llm         base, http, groq, gemini, ollama, mock/scripted, router, factory, prompts
app/api         FastAPI app, routes (health, documents, chat, sources, admin)
app/scripts     ingest_pdf CLI
frontend/       Next.js App Router UI (sidebar, streaming chat, citation badges, source inspector)
tests/          router, chunking, cleaning/structure, retrieval, API
docs/           free_tier_audit.md
```

## Known limitations

* Multi-column layouts rely on the PDF's content order; check a dry-run sample of your book.
* Tables are extracted as text lines (no table reconstruction); figures are ignored.
* "PDF p." is the physical PDF page, which may differ from the printed page number.
* Follow-up questions are handled by a heuristic (previous question is prepended for short or pronoun-heavy
  follow-ups) so that no LLM quota is spent on query rewriting.
* Conversation history lives in the browser's localStorage; there are no accounts.
* This is a study aid, not medical advice. The prompt asks the model to cite every claim and to say when the
  text does not cover a question, but models can still misread a passage: always check the cited source.
