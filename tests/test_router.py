"""LLM router: primary success, rate-limit failover, streaming error recovery, exhaustion, timeouts,
cooldowns, and provider wire-format handling with mocked HTTP callers (no network, no pytest plugins)."""
from __future__ import annotations

import asyncio
import json

from app.llm.base import ChatMessage
from app.llm.gemini import GeminiProvider, to_gemini_payload
from app.llm.groq import GroqProvider
from app.llm.http import StreamTimeouts, UpstreamConnectionError, UpstreamHTTPError, UpstreamTimeout
from app.llm.mock import Fail, ScriptedProvider, Sleep
from app.llm.ollama import OllamaProvider
from app.llm.router import LLMRouter

MSGS = [ChatMessage("system", "sys"), ChatMessage("user", "hello")]
LONG = "x" * 200  # exceeds the commit threshold on its own


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


async def collect(router: LLMRouter, **kw):
    return [ev async for ev in router.stream(MSGS, **kw)]


def types(events):
    return [e.type for e in events]


def text(events):
    return "".join(e.data["text"] for e in events if e.type == "token")


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- router behaviour
def test_primary_success_never_touches_secondary():
    groq = ScriptedProvider("groq", "llama", [["Hello ", "world ", LONG]])
    gemini = ScriptedProvider("gemini", "flash", [["unused"]])
    events = run(collect(LLMRouter([groq, gemini], commit_chars=50)))
    assert types(events)[0] == "provider" and events[0].data["provider"] == "groq"
    assert types(events)[-1] == "done"
    assert "restart" not in types(events)
    assert text(events) == "Hello world " + LONG
    assert groq.calls == 1 and gemini.calls == 0


def test_short_answer_is_committed_when_stream_ends():
    groq = ScriptedProvider("groq", "llama", [["Hi."]])
    events = run(collect(LLMRouter([groq], commit_chars=500)))
    assert types(events) == ["provider", "token", "done"]


def test_rate_limit_fails_over_before_any_token_is_shown():
    groq = ScriptedProvider("groq", "llama", [[Fail("rate_limit", 429, retry_after=30)]])
    gemini = ScriptedProvider("gemini", "flash", [["From ", "Gemini. ", LONG]])
    events = run(collect(LLMRouter([groq, gemini], commit_chars=50)))
    assert types(events) == ["restart", "provider", "token", "token", "token", "done"]
    restart = events[0].data
    assert restart["reason"] == "rate_limit" and restart["from"] == "groq/llama" and restart["to"] == "gemini/flash"
    assert text(events).startswith("From Gemini.")
    assert events[-1].data["provider"] == "gemini" and events[-1].data["restarts"] == 1
    assert groq.calls == 1 and gemini.calls == 1


def test_error_before_commit_discards_buffered_tokens():
    groq = ScriptedProvider("groq", "llama", [["partial ", "garbage ", Fail("unavailable", 503)]])
    gemini = ScriptedProvider("gemini", "flash", [["clean ", "answer ", LONG]])
    events = run(collect(LLMRouter([groq, gemini], commit_chars=100)))
    out = text(events)
    assert "garbage" not in out and "partial" not in out
    assert out.startswith("clean answer")
    restart = next(e for e in events if e.type == "restart")
    assert restart.data["reason"] == "unavailable"
    assert restart.data["discarded_chars"] == len("partial garbage ")
    # nothing from the failed provider was emitted before the restart notice
    assert all(e.type != "token" for e in events[: events.index(restart)])


def test_error_after_commit_is_reported_as_partial_not_failed_over():
    groq = ScriptedProvider("groq", "llama", [[LONG, "more ", Fail("unavailable", 503)]])
    gemini = ScriptedProvider("gemini", "flash", [["must not run"]])
    events = run(collect(LLMRouter([groq, gemini], commit_chars=50)))
    assert types(events)[-1] == "error"
    err = events[-1].data
    assert err["code"] == "stream_interrupted" and err["partial"] is True
    assert "restart" not in types(events)
    assert text(events).startswith("x" * 200)
    assert gemini.calls == 0


def test_all_providers_exhausted_returns_terminal_error_event():
    p1 = ScriptedProvider("groq", "a", [[Fail("rate_limit", 429, retry_after=45)]])
    p2 = ScriptedProvider("gemini", "b", [[Fail("rate_limit", 429, retry_after=90)]])
    p3 = ScriptedProvider("ollama", "c", [[Fail("connection")]])
    clock = FakeClock()
    events = run(collect(LLMRouter([p1, p2, p3], clock=clock)))
    assert types(events) == ["restart", "restart", "error"]
    err = events[-1].data
    assert err["code"] == "all_providers_failed" and err["partial"] is False
    assert [a["kind"] for a in err["attempts"]] == ["rate_limit", "rate_limit", "connection"]
    assert err["retry_after_s"] == 30.0  # connection cooldown is the shortest
    assert (p1.calls, p2.calls, p3.calls) == (1, 1, 1)


def test_exhausted_message_names_the_real_cause_not_just_rate_limits():
    groq = ScriptedProvider("groq", "a", [[Fail("rate_limit", 429, retry_after=3000)]])
    gemini = ScriptedProvider("gemini", "b", [[Fail("not_found", 404)]])
    router = LLMRouter([groq, gemini], clock=FakeClock())
    err = run(collect(router))[-1].data
    assert "groq/a rate-limited" in err["message"] and "gemini/b model not found" in err["message"]
    assert [(a["kind"], a.get("status")) for a in err["attempts"]] == [("rate_limit", 429), ("not_found", 404)]
    # next request: nothing is even tried, but the *original* reasons are still reported
    again = run(collect(router))[-1].data
    assert (groq.calls, gemini.calls) == (1, 1)
    assert again["attempts"][0]["kind"] == "cooldown" and again["attempts"][0]["was"] == "rate_limit"
    assert "groq/a rate-limited, skipped" in again["message"] and "gemini/b model not found, skipped" in again["message"]
    st = {s["provider"]: s for s in router.status()}
    assert st["gemini"]["last_status"] == 404 and "scripted failure" in st["gemini"]["last_detail"]


def test_first_token_timeout_triggers_failover():
    slow = ScriptedProvider("groq", "slow", [[Sleep(5), "never"]])
    fast = ScriptedProvider("gemini", "fast", [["ok ", LONG]])
    events = run(collect(LLMRouter([slow, fast], commit_chars=50, first_token_timeout=0.05)))
    assert next(e for e in events if e.type == "restart").data["reason"] == "timeout"
    assert text(events).startswith("ok ")


def test_empty_response_counts_as_failure():
    empty = ScriptedProvider("groq", "a", [[]])
    ok = ScriptedProvider("gemini", "b", [["fine"]])
    events = run(collect(LLMRouter([empty, ok])))
    assert next(e for e in events if e.type == "restart").data["reason"] == "empty"
    assert text(events) == "fine"


def test_cooldown_skips_failed_provider_on_next_request_then_recovers():
    clock = FakeClock()
    groq = ScriptedProvider("groq", "a", [[Fail("rate_limit", 429, retry_after=60)], ["back ", LONG]])
    gemini = ScriptedProvider("gemini", "b", [["g1 ", LONG], ["g2 ", LONG], ["g3 ", LONG]])
    router = LLMRouter([groq, gemini], commit_chars=50, clock=clock)

    first = run(collect(router))
    assert groq.calls == 1 and "restart" in types(first)

    second = run(collect(router))  # groq is cooling down: skipped silently, no restart notice
    assert groq.calls == 1 and gemini.calls == 2
    assert "restart" not in types(second) and text(second).startswith("g2")
    status = {s["provider"]: s for s in router.status()}
    assert status["groq"]["available"] is False and status["groq"]["cooldown_remaining_s"] == 60.0

    clock.t += 61
    third = run(collect(router))
    assert groq.calls == 2 and text(third).startswith("back")


def test_unexpected_exception_in_provider_is_contained():
    class Boom(ScriptedProvider):
        async def stream(self, messages, *, temperature, max_tokens):
            raise RuntimeError("bug")
            yield ""  # pragma: no cover

    ok = ScriptedProvider("gemini", "b", [["fine"]])
    events = run(collect(LLMRouter([Boom("groq", "a", [[]]), ok])))
    assert next(e for e in events if e.type == "restart").data["reason"] == "internal"
    assert text(events) == "fine"


def test_no_providers_configured():
    events = run(collect(LLMRouter([])))
    assert types(events) == ["error"] and events[0].data["code"] == "all_providers_failed"


# ----------------------------------------------------------- provider wire formats
class FakeStreamer:
    """Mocked HTTP caller: replays canned lines, or raises a canned transport/HTTP error."""

    def __init__(self, lines=(), error=None, fail_after=None):
        self.lines, self.error, self.fail_after = list(lines), error, fail_after
        self.requests = []

    async def stream_lines(self, url, *, headers, json_body, timeouts):
        self.requests.append({"url": url, "headers": headers, "body": json_body})
        if self.error is not None and self.fail_after is None:
            raise self.error
        for i, line in enumerate(self.lines):
            if self.fail_after is not None and i == self.fail_after:
                raise self.error
            yield line


def _collect_provider(provider):
    async def go():
        return [t async for t in provider.stream(MSGS, temperature=0.1, max_tokens=50)]

    return run(go())


def _error_kind(provider):
    try:
        _collect_provider(provider)
    except Exception as exc:  # noqa: BLE001
        return exc.kind, getattr(exc, "retry_after", None)
    raise AssertionError("expected a ProviderError")


def sse_data(obj):
    return "data: " + json.dumps(obj)


def test_groq_streams_openai_sse_and_sends_bearer_auth():
    lines = [
        sse_data({"choices": [{"delta": {"role": "assistant"}}]}),
        "",
        sse_data({"choices": [{"delta": {"content": "Hel"}}]}),
        ": keep-alive",
        sse_data({"choices": [{"delta": {"content": "lo"}}]}),
        sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    http = FakeStreamer(lines)
    p = GroqProvider(api_key="k", model="m", http=http)
    assert _collect_provider(p) == ["Hel", "lo"]
    req = http.requests[0]
    assert req["url"].endswith("/chat/completions") and req["headers"]["Authorization"] == "Bearer k"
    assert req["body"]["stream"] is True and req["body"]["model"] == "m"


def test_groq_http_429_maps_to_rate_limit_with_retry_after():
    p = GroqProvider(api_key="k", model="m", http=FakeStreamer(error=UpstreamHTTPError(429, "slow down", 17.0)))
    assert _error_kind(p) == ("rate_limit", 17.0)


def test_groq_status_classification():
    expect = {503: "unavailable", 502: "unavailable", 401: "auth", 404: "not_found", 400: "bad_request"}
    for status, kind in expect.items():
        p = GroqProvider(api_key="k", model="m", http=FakeStreamer(error=UpstreamHTTPError(status, "")))
        assert _error_kind(p)[0] == kind


def test_timeout_and_connection_errors_are_classified():
    assert _error_kind(GroqProvider(api_key="k", model="m", http=FakeStreamer(error=UpstreamTimeout("t"))))[0] == "timeout"
    assert (
        _error_kind(GroqProvider(api_key="k", model="m", http=FakeStreamer(error=UpstreamConnectionError("c"))))[0]
        == "connection"
    )


def test_groq_mid_stream_error_object_is_a_rate_limit():
    lines = [
        sse_data({"choices": [{"delta": {"content": "a"}}]}),
        sse_data({"error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}}),
    ]
    assert _error_kind(GroqProvider(api_key="k", model="m", http=FakeStreamer(lines)))[0] == "rate_limit"


def test_gemini_payload_and_stream_parsing():
    msgs = [ChatMessage("system", "S"), ChatMessage("user", "a"), ChatMessage("user", "b"), ChatMessage("assistant", "c")]
    payload = to_gemini_payload(msgs, temperature=0.2, max_tokens=10, model="gemini-2.5-flash")
    assert payload["systemInstruction"]["parts"][0]["text"] == "S"
    assert [c["role"] for c in payload["contents"]] == ["user", "model"]  # consecutive user turns merged
    assert payload["contents"][0]["parts"][0]["text"] == "a\n\nb"
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    lines = [
        sse_data({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]}),
        sse_data({"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}),
        sse_data({"candidates": [{"content": {"parts": [{"text": "there"}]}, "finishReason": "STOP"}]}),
    ]
    http = FakeStreamer(lines)
    p = GeminiProvider(api_key="g", model="gemini-2.5-flash", http=http)
    assert _collect_provider(p) == ["Hello ", "there"]
    req = http.requests[0]
    assert req["headers"]["x-goog-api-key"] == "g" and "key=" not in req["url"]
    assert "streamGenerateContent?alt=sse" in req["url"]


def test_gemini_quota_error_reads_retry_delay_from_body():
    body = json.dumps({"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "32s"}]}})
    p = GeminiProvider(api_key="g", model="m", http=FakeStreamer(error=UpstreamHTTPError(429, body)))
    assert _error_kind(p) == ("rate_limit", 32.0)


def test_gemini_safety_block_is_reported():
    lines = [sse_data({"candidates": [{"finishReason": "SAFETY"}]})]
    assert _error_kind(GeminiProvider(api_key="g", model="m", http=FakeStreamer(lines)))[0] == "blocked"


def test_ollama_ndjson_and_connection_refused():
    lines = [
        json.dumps({"message": {"content": "Hi "}, "done": False}),
        json.dumps({"message": {"content": "there"}, "done": False}),
        json.dumps({"done": True}),
    ]
    assert _collect_provider(OllamaProvider(model="m", http=FakeStreamer(lines))) == ["Hi ", "there"]
    assert _error_kind(OllamaProvider(model="m", http=FakeStreamer(error=UpstreamConnectionError("refused"))))[0] == "connection"


def test_end_to_end_failover_across_real_provider_classes_with_mocked_http():
    groq = GroqProvider(api_key="k", model="llama", http=FakeStreamer(error=UpstreamHTTPError(429, "", 20.0)))
    gem_lines = [sse_data({"candidates": [{"content": {"parts": [{"text": "Answer from Gemini. " + LONG}]}}]})]
    gemini = GeminiProvider(api_key="g", model="flash", http=FakeStreamer(gem_lines))
    events = run(collect(LLMRouter([groq, gemini], commit_chars=50)))
    assert types(events)[0] == "restart" and events[0].data["reason"] == "rate_limit"
    assert text(events).startswith("Answer from Gemini.") and types(events)[-1] == "done"


def test_midstream_http_drop_before_commit_fails_over():
    lines = [sse_data({"choices": [{"delta": {"content": "par"}}]}), sse_data({"choices": [{"delta": {"content": "tial"}}]})]
    groq = GroqProvider(
        api_key="k", model="m", http=FakeStreamer(lines, error=UpstreamConnectionError("reset"), fail_after=1)
    )
    ok = ScriptedProvider("gemini", "b", [["clean ", LONG]])
    events = run(collect(LLMRouter([groq, ok], commit_chars=100)))
    assert "par" not in text(events) and text(events).startswith("clean")
