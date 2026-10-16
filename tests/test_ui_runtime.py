"""The sync bridge used by the Streamlit app, with fakes (no qdrant, no model, no network)."""
from __future__ import annotations

import types

from app.llm.mock import MockProvider
from app.llm.router import LLMRouter
from app.retrieval.models import Candidate, RetrievalResult
from app.ui_runtime import Runtime

SETTINGS = types.SimpleNamespace(
    max_history_messages=4, llm_max_input_tokens=4500, llm_temperature=0.1, llm_max_output_tokens=100,
)


class FakeRetriever:
    def __init__(self, confident=True):
        self.confident = confident

    async def retrieve(self, query, *, document_id=None):
        cand = Candidate("c1", {"text": "The heart has four chambers.", "page_start": 5, "page_end": 5}, 0.1, dense_score=0.8)
        return RetrievalResult([cand], self.confident, 0.8 if self.confident else 0.2, "ok", {})


def events(rt, q="heart?"):
    return list(rt.stream(q, []))


def test_streams_sources_tokens_and_done_in_order():
    rt = Runtime(FakeRetriever(), LLMRouter([MockProvider()], commit_chars=10), SETTINGS)
    evs = events(rt)
    names = [e for e, _ in evs]
    assert names[0] == "sources" and names[-1] == "done" and "provider" in names
    assert "four chambers" in "".join(d["text"] for e, d in evs if e == "token")
    assert evs[0][1]["sources"][0]["ref"] == "S1"


def test_low_confidence_skips_the_llm():
    mock = MockProvider()
    rt = Runtime(FakeRetriever(confident=False), LLMRouter([mock]), SETTINGS)
    evs = events(rt)
    assert [e for e, _ in evs] == ["sources", "token", "done"] and evs[-1][1]["reason"] == "insufficient_evidence"


def test_runtime_is_reusable_across_requests_and_reports_status():
    rt = Runtime(FakeRetriever(), LLMRouter([MockProvider()]), SETTINGS)
    assert events(rt)[-1][0] == "done" and events(rt)[-1][0] == "done"
    assert rt.provider_status()[0]["successes"] == 2


def test_no_providers_yields_a_clear_error():
    rt = Runtime(FakeRetriever(), LLMRouter([]), SETTINGS)
    last = events(rt)[-1]
    assert last[0] == "error" and "No language model is configured" in last[1]["message"]
