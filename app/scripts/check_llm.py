"""Probe every provider in LLM_CHAIN with a tiny request and print the real failure reason.

    python -m app.scripts.check_llm              # probe each provider
    python -m app.scripts.check_llm --gemini-models   # also list the Gemini models your key can call

Uses the same .env, the same provider classes and the same error classification as the API, so what you see
here is exactly what the router sees. Sends one short fixed prompt (no textbook content).
"""
from __future__ import annotations

import argparse
import asyncio
import time

from app.core.config import get_settings
from app.llm.base import ChatMessage, ProviderError
from app.llm.factory import _secret, build_providers, parse_chain
from app.llm.http import HttpxStreamer

PROMPT = [ChatMessage("user", "Reply with the single word: ok")]


async def probe(provider) -> str:
    t0 = time.perf_counter()
    try:
        out = ""
        async for tok in provider.stream(PROMPT, temperature=0.0, max_tokens=8):
            out += tok
            if len(out) >= 2:
                break
        return f"OK     {provider.label}  ({(time.perf_counter() - t0) * 1000:.0f} ms) -> {out.strip()[:20]!r}"
    except ProviderError as exc:
        wait = f", retry_after={exc.retry_after:.0f}s" if exc.retry_after else ""
        return f"FAIL   {provider.label}  kind={exc.kind} status={exc.status}{wait}\n         {exc}"


def list_gemini_models(key: str) -> None:
    import httpx

    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": key}, timeout=15)
    if r.status_code != 200:
        print(f"Gemini model list failed: HTTP {r.status_code} {r.text[:300]}")
        return
    names = sorted(
        m["name"].removeprefix("models/")
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )
    print("Gemini models usable with generateContent (put one in LLM_CHAIN as gemini:<name>):")
    for n in names:
        print("  ", n)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini-models", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    print("LLM_CHAIN =", s.llm_chain)
    configured = [f"{n}:{m}" for n, m in parse_chain(s.llm_chain)]
    http = HttpxStreamer()
    providers = build_providers(s, http)
    skipped = len(configured) - len(providers)
    if skipped:
        print(f"{skipped} entr{'y' if skipped == 1 else 'ies'} skipped (missing API key) - see log lines above")
    for p in providers:
        print(await probe(p))
    if args.gemini_models and _secret(s.gemini_api_key):
        list_gemini_models(_secret(s.gemini_api_key))
    await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
