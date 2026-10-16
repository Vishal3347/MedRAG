"""API integration tests (FastAPI TestClient) with fake retriever/store and a scripted LLM router."""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.api.deps import Container  # noqa: E402
from app.api.main import create_app  # noqa: E402
from app.core.ratelimit import SlidingWindowLimiter  # noqa: E402
from app.llm.mock import Fail, ScriptedProvider  # noqa: E402
from app.llm.router import LLMRouter  # noqa: E402
from app.retrieval.models import Candidate, RetrievalResult  # noqa: E402

CHUNK_ID = str(uuid.uuid4())
PAYLOAD = {
    "document_id": "doc_1", "document_title": "Clinical Medicine", "chapter": "Chapter 4: Renal", "section_heading": "AKI",
    "page_start": 120, "page_end": 121, "text": "## AKI\nAcute kidney injury is a rapid decline in GFR.", "content_hash": "h",
}


class FakeRetriever:
    candidates, top_k, min_dense_score = 20, 5, 0.45

    def __init__(self, confident=True):
        self.confident = confident
        self.queries = []

    async def retrieve(self, query, *, document_id=None):
        self.queries.append(query)
        chunk = Candidate(CHUNK_ID, dict(PAYLOAD), 0.03, dense_score=0.82)
        return RetrievalResult([chunk], self.confident, 0.82 if self.confident else 0.2, "ok" if self.confident else "low_similarity")


class FakeStore:
    def list_documents(self):
        return [{"document_id": "doc_1", "title": "Clinical Medicine", "chunks": 1, "page_start": 1, "page_end": 900, "chapters": 40}]

    def get_chunk(self, chunk_id):
        return {"id": chunk_id, **PAYLOAD} if chunk_id == CHUNK_ID else None

    def ping(self):
        return True

    def points_count(self):
        return 1


def make_client(*, confident=True, provider=None, admin_key="secret", rate=100):
    provider = provider or ScriptedProvider("groq", "llama", [["Acute kidney injury is a fast fall in GFR [S1]. ", "x" * 150]])
    settings = SimpleNamespace(
        admin_api_key=SecretStr(admin_key), cors_origin_list=["http://localhost:3000"], log_level="WARNING", log_json=False,
        llm_temperature=0.1, llm_max_output_tokens=100, llm_max_input_tokens=4500, max_history_messages=8, trust_proxy=False,
    )
    retriever = FakeRetriever(confident)
    container = Container(settings, retriever, LLMRouter([provider], commit_chars=40), FakeStore(), limiter=SlidingWindowLimiter(rate))
    return TestClient(create_app(container)), retriever, provider


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((name, json.loads(data) if data else None))
    return events


def test_health_ready_and_request_id_header():
    client, _, _ = make_client()
    r = client.get("/health", headers={"X-Request-ID": "abcdef123456"})
    assert r.status_code == 200 and r.json() == {"status": "ok"} and r.headers["x-request-id"] == "abcdef123456"
    assert client.get("/health").headers["x-request-id"]
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["indexed_chunks"] == 1


def test_documents_and_sources_endpoints():
    client, _, _ = make_client()
    assert client.get("/api/v1/documents").json()["documents"][0]["title"] == "Clinical Medicine"
    assert client.get(f"/api/v1/sources/{CHUNK_ID}").json()["text"].endswith("decline in GFR.")
    assert client.get(f"/api/v1/sources/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/v1/sources/not-a-uuid").status_code == 404


def test_chat_stream_emits_sources_tokens_and_done():
    client, retriever, provider = make_client()
    r = client.post("/api/v1/chat/stream", json={"question": "What is AKI?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[:2] == ["meta", "sources"] and names[-1] == "done"
    assert "provider" in names and "token" in names
    sources = events[1][1]
    assert sources["confident"] is True and sources["sources"][0]["ref"] == "S1"
    assert sources["sources"][0]["chunk_id"] == CHUNK_ID and sources["sources"][0]["snippet"].startswith("AKI Acute")
    assert "[S1]" in "".join(d["text"] for n, d in events if n == "token")
    assert provider.calls == 1 and retriever.queries == ["What is AKI?"]


def test_insufficient_evidence_never_calls_the_llm():
    client, _, provider = make_client(confident=False)
    events = parse_sse(client.post("/api/v1/chat/stream", json={"question": "Who won the world cup?"}).text)
    assert events[1][1]["confident"] is False
    assert events[-1][0] == "done" and events[-1][1]["reason"] == "insufficient_evidence"
    assert provider.calls == 0
    assert "couldn't find enough support" in "".join(d["text"] for n, d in events if n == "token")


def test_all_providers_failing_yields_an_error_event_not_an_http_error():
    provider = ScriptedProvider("groq", "llama", [[Fail("rate_limit", 429)]])
    client, _, _ = make_client(provider=provider)
    events = parse_sse(client.post("/api/v1/chat/stream", json={"question": "What is AKI?"}).text)
    assert events[-1][0] == "error" and events[-1][1]["code"] == "all_providers_failed"


def test_request_validation():
    client, _, _ = make_client()
    assert client.post("/api/v1/chat/stream", json={"question": "   "}).status_code == 422
    assert client.post("/api/v1/chat/stream", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/v1/chat/stream", json={"question": "ok", "document_id": "bad id!"}).status_code == 422


def test_rate_limit_returns_429_with_retry_after():
    client, _, _ = make_client(rate=1)
    assert client.post("/api/v1/chat/stream", json={"question": "first"}).status_code == 200
    second = client.post("/api/v1/chat/stream", json={"question": "second"})
    assert second.status_code == 429 and int(second.headers["retry-after"]) >= 1


def test_admin_endpoints_require_the_key():
    client, _, _ = make_client()
    assert client.get("/api/v1/admin/status").status_code == 401
    assert client.get("/api/v1/admin/status", headers={"X-Admin-Key": "wrong"}).status_code == 401
    ok = client.get("/api/v1/admin/status", headers={"X-Admin-Key": "secret"})
    assert ok.status_code == 200 and ok.json()["providers"][0]["provider"] == "groq"
    dbg = client.get("/api/v1/admin/retrieve", params={"q": "aki"}, headers={"X-Admin-Key": "secret"})
    assert dbg.status_code == 200 and dbg.json()["chunks"][0]["dense"] == 0.82
    assert client.post("/api/v1/admin/providers/reset", headers={"X-Admin-Key": "secret"}).status_code == 200


def test_admin_is_disabled_when_no_key_is_configured():
    client, _, _ = make_client(admin_key="")
    assert client.get("/api/v1/admin/status", headers={"X-Admin-Key": ""}).status_code == 503
