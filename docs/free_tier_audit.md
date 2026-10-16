# Free-tier audit

> **Read this first.** This file was written in a sandbox with no internet access, so **none of the figures
> below were verified live**. They are the last values I knew (2025, plus what I believe still held in
> mid-2026). Free tiers change often, and Gemini's free quotas in particular were cut more than once.
> Before you rely on any number, open the "Where to check" link and update the table. The architecture does not
> depend on exact numbers: every external dependency has a fallback and a cooldown-based circuit breaker.

## Update (Oct 2026): why a single question can exhaust every provider

* Run `python -m app.scripts.check_llm --gemini-models` to see the *real* status of each provider and which Gemini
  model ids your key may call. A secondary source (GitHub issue robhunter/agentdeals#2017, not Google's own page)
  says Gemini 2.5 models have been restricted to projects that used them before 2026-09-18 and that free-tier
  quotas are no longer published (they are shown in AI Studio). Treat `gemini-2.5-flash` as possibly dead for a
  new key; a 404/403 shows up as `not_found`/`auth` with a 10-minute cooldown.
* Groq's 8B model allows only 6,000 tokens/minute on the free plan, so the prompt is now capped
  (`LLM_MAX_INPUT_TOKENS`) and history is trimmed (`MAX_HISTORY_MESSAGES=4`).

## Summary

| Service | Role | Cost | Main risk | Mitigation in this repo |
|---|---|---|---|---|
| Qdrant Cloud free cluster | Vector DB (dense + sparse) | $0, no card (as last known) | Idle clusters get suspended, then deleted; small RAM | Re-ingest is idempotent (one command); keep-alive ping to `/ready`; local `qdrant` docker profile |
| Groq API (free plan) | Primary LLM | $0 | Per-model, per-minute and per-**day** token caps; model deprecations | `LLM_CHAIN` lists two Groq models (separate quotas); automatic failover; cooldown honours `Retry-After` |
| Google Gemini API (free tier) | Secondary LLM | $0 | Low daily request cap; limits change; free-tier prompts may be used to improve Google products | Failover chain; request rate limiter; safety thresholds relaxed for medical text |
| Ollama (local) | Tertiary LLM | $0 (your hardware) | Needs a machine with RAM; slow on CPU | Optional; connection errors fail over instantly |
| sentence-transformers (BGE small) | Embeddings | $0, runs locally | ~0.5-1 GB RAM with torch | CPU torch in Docker; model baked into the image |
| Backend hosting | FastAPI + embedder | $0 options only | Free web tiers often have 512 MB RAM; cold starts | See "Hosting" below |
| Frontend hosting | Next.js | $0 | None significant | Static-friendly; API URL set at build time |

## Qdrant Cloud

* Free tier (last known): one single-node cluster, about 1 GB RAM / 4 GB disk, no credit card.
* This workload: ~1,000 pages is roughly 2,000-3,500 chunks. 384-dim float32 vectors are ~1.5 KB each, plus
  sparse vectors and ~3 KB text payloads, so the collection is **well under 100 MB**. Plenty of headroom.
* **Idle suspension:** free clusters are suspended after a period of inactivity and eventually deleted if not
  reactivated (I believe about a week, then a longer grace period; verify). If you deploy publicly, ping `/ready`
  from a free cron (GitHub Actions schedule, cron-job.org) at least a few times a week. If the cluster is ever
  lost, `python -m app.scripts.ingest_pdf --pdf-path book.pdf` rebuilds everything.
* Where to check: https://qdrant.tech/pricing/ and the "Free tier" section of the Cloud docs.

## Groq

* Free plan limits are per model and per organisation. Last known values for the models in the default chain
  (verify in the console under Settings > Limits):

  | Model | Requests/min | Requests/day | Tokens/min | Tokens/day |
  |---|---|---|---|---|
  | `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
  | `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | 500,000 |

* **Capacity math (if those numbers still hold).** One question sends roughly 5 passages x ~600 tokens plus the
  system prompt and history, about 3.5-4.5k input tokens, and up to 900 output tokens: **~5k tokens per
  question**. That is only **~20 questions per day on the 70B model** before its daily token cap, after which the
  router moves to the 8B model (~100 questions/day), then Gemini. For heavier use reduce `CONTEXT_CHUNKS` to 4,
  lower `LLM_MAX_OUTPUT_TOKENS`, or put Gemini first in `LLM_CHAIN`.
* `mixtral-8x7b-32768`, named in the original spec, was deprecated by Groq in 2025 as far as I know, so it is not
  in the defaults. The model list changes; check https://console.groq.com/docs/models and
  https://console.groq.com/docs/rate-limits.
* Failure signatures handled: HTTP 429 (per-minute and per-day), 503/5xx, timeouts, mid-stream error objects.

## Google Gemini

* Free tier (last known for 2.5 Flash, mid-2025): roughly 10 requests/min, ~250 requests/day, ~250k tokens/min.
  I believe these were reduced later in 2025 and may differ again now. Check https://ai.google.dev/gemini-api/docs/rate-limits
  and https://ai.google.dev/gemini-api/docs/pricing.
* `gemini-1.5-flash` (named in the spec) has been retired, so the default is `gemini-2.5-flash`. For 2.5 Flash
  the client disables "thinking" (`thinkingBudget: 0`) so quota goes to the answer.
* Free-tier content may be used by Google to improve its products. Your textbook excerpts are sent to the
  provider with every question. Check the licence of the book, and do not send patient data.
* 429 bodies contain `retryDelay`; the router parses it and sits the model out for exactly that long.
* The 2.5 Flash Lite variant (separate quota, higher caps) is a good extra entry in the chain:
  `gemini:gemini-2.5-flash-lite` (verify the model id).

## Ollama (optional tertiary)

* Free, local. A 7-8B model (`llama3.1:8b`, `qwen2.5:7b`) needs roughly 6-8 GB RAM and is slow on CPU, but works
  as a last resort. Set `OLLAMA_BASE_URL`. If it is not running, the connection error is classified in
  milliseconds and the request ends with a clear "all providers unavailable" message.
* For a demo without any keys at all, set `LLM_CHAIN=mock:echo`.

## Embeddings

* `BAAI/bge-small-en-v1.5`: 384 dims, 512-token limit, ~130 MB download, CPU is fine (a 1,000-page book embeds in
  a few minutes on a laptop). `all-MiniLM-L6-v2` is smaller but has only a 256-token limit, so with 600-token
  chunks it would only embed the first ~40% of each chunk. Prefer BGE.

## Hosting (all unverified, pick one that fits)

* The backend needs ~1 GB RAM (torch + model). **A 512 MB free web tier will probably run out of memory.**
  Candidates with a free option: Hugging Face Spaces (Docker Space, CPU basic; sleeps when idle), Oracle Cloud
  "Always Free" VMs, or your own machine behind a tunnel. Free instances that sleep add a cold start of tens of
  seconds on the first request.
* The frontend is a standard Next.js app: Vercel Hobby, Cloudflare Pages or Netlify free plans all work. Set
  `NEXT_PUBLIC_API_BASE_URL` at build time and add the frontend origin to `CORS_ORIGINS` on the backend.

## Fallback strategy (what happens when something runs out)

1. Groq 70B returns 429 -> router records a cooldown from `Retry-After`, silently continues with Groq 8B.
2. Groq exhausted -> Gemini. Gemini exhausted -> Ollama (if running).
3. Everything exhausted -> the user gets one clear message and a retry hint; the next request skips providers that
   are still cooling down, so no quota is burned on doomed calls.
4. Retrieval is independent of the LLM: the user still sees the closest passages with citations even when no
   model is available (sources are sent before generation starts).
5. Qdrant down -> `/ready` fails and chat returns a `retrieval_failed` error; nothing is guessed.
