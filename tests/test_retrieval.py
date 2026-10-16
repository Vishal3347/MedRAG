from __future__ import annotations

import asyncio

from app.llm.prompts import SYSTEM_PROMPT, build_messages, estimate_tokens, format_context
from app.retrieval.fusion import dedupe, rrf_fuse
from app.retrieval.models import Hit
from app.retrieval.query import build_retrieval_query
from app.retrieval.retriever import HybridRetriever
from app.retrieval.sparse import encode_document, encode_query, term_index, tokenize
from app.core.ratelimit import SlidingWindowLimiter


# ------------------------------------------------------------------------- RRF
def test_rrf_rewards_agreement_between_rankers():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    order = [d for d, _ in fused]
    assert order[:2] == ["a", "c"] and set(order) == {"a", "b", "c", "d"}
    expected_a = 1 / 61 + 1 / 62
    assert abs(dict(fused)["a"] - expected_a) < 1e-12


def test_rrf_ties_are_deterministic_and_weights_work():
    assert [d for d, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]
    heavy = rrf_fuse([["x"], ["y"]], weights=[1.0, 3.0])
    assert heavy[0][0] == "y"


def test_dedupe_drops_exact_and_near_duplicates_keeping_the_best_ranked():
    base = "the quick brown fox jumps over the lazy dog while the patient receives fluids " * 3
    items = [
        {"id": 1, "text": base, "h": "h1"},
        {"id": 2, "text": base + " extra", "h": "h2"},  # near duplicate
        {"id": 3, "text": "completely different passage about renal tubular acidosis and bicarbonate", "h": "h3"},
        {"id": 4, "text": "another unique text", "h": "h3"},  # same hash as 3
    ]
    kept = dedupe(items, text_of=lambda i: i["text"], hash_of=lambda i: i["h"])
    assert [i["id"] for i in kept] == [1, 3]


# ---------------------------------------------------------------------- sparse
def test_tokenizer_keeps_medical_compounds_numbers_and_negations():
    toks = tokenize("Beta-blockers are NOT indicated; K+ 5.5 mg/dL without insufficiency.")
    for expected in ("beta-blockers", "beta", "blockers", "not", "5.5", "mg/dl", "mg", "dl", "without"):
        assert expected in toks
    assert "are" not in toks and "the" not in tokenize("the")


def test_sparse_encoding_is_deterministic_sorted_and_matches_queries():
    idx, val = encode_document("Metformin is first-line therapy for type 2 diabetes. Metformin lowers glucose.")
    assert idx == sorted(idx) and len(idx) == len(val) and all(v > 0 for v in val)
    assert encode_document("same text") == encode_document("same text")
    q_idx, q_val = encode_query("metformin therapy")
    assert set(q_idx) <= set(idx) and q_val == [1.0] * len(q_idx)
    weights = dict(zip(idx, val))
    assert weights[term_index("metformin")] > weights[term_index("glucose")]  # tf=2 beats tf=1


def test_empty_document_still_encodes():
    idx, val = encode_document("the of and")
    assert len(idx) == 1 and val == [1.0]


# ------------------------------------------------------------------- retriever
class FakeEmbedder:
    def embed_query(self, text):
        return [0.0, 1.0]


class FakeStore:
    def __init__(self, dense, sparse, sparse_error=None):
        self._dense, self._sparse, self._sparse_error = dense, sparse, sparse_error
        self.dense_calls = self.sparse_calls = 0

    def dense_search(self, vector, limit, document_id=None):
        self.dense_calls += 1
        return self._dense[:limit]

    def sparse_search(self, indices, values, limit, document_id=None):
        self.sparse_calls += 1
        if self._sparse_error:
            raise self._sparse_error
        return self._sparse[:limit]


def hit(i, score, text=None):
    return Hit(f"id{i}", score, {"text": text or f"unique passage number {i} " * 6, "content_hash": f"h{i}", "page_start": i, "page_end": i})


def retrieve(store, **kw):
    r = HybridRetriever(store, FakeEmbedder(), candidates=20, top_k=5, **kw)
    return asyncio.run(r.retrieve("what is the first-line therapy for hypertension"))


def test_hybrid_retrieval_fuses_pools_and_returns_top_five():
    dense = [hit(i, 0.80 - i * 0.01) for i in range(1, 21)]
    sparse = [hit(i, 30 - i) for i in (7, 3, 1, 25, 26, 27)]
    result = retrieve(FakeStore(dense, sparse))
    ids = [c.id for c in result.chunks]
    assert len(ids) == 5 and result.confident and result.reason == "ok"
    assert ids[0] in ("id1", "id3", "id7")  # candidates found by both retrievers float to the top
    assert {"id1", "id3", "id7"} <= set(ids)
    top = result.chunks[0]
    assert top.dense_score is not None and top.sparse_score is not None
    assert result.stats["dense_candidates"] == 20 and result.stats["sparse_candidates"] == 6


def test_low_similarity_marks_evidence_insufficient_but_still_returns_nearest_chunks():
    result = retrieve(FakeStore([hit(1, 0.31), hit(2, 0.28)], [hit(2, 5.0)]), min_dense_score=0.45)
    assert not result.confident and result.reason == "low_similarity" and result.chunks


def test_no_results_is_reported():
    result = retrieve(FakeStore([], []))
    assert not result.confident and result.reason == "no_results" and result.chunks == []


def test_sparse_failure_degrades_to_dense_only():
    result = retrieve(FakeStore([hit(1, 0.7), hit(2, 0.6)], [], sparse_error=RuntimeError("boom")))
    assert result.confident and [c.id for c in result.chunks] == ["id1", "id2"]


def test_duplicate_chunks_are_removed_from_context():
    same = "identical boilerplate paragraph about informed consent and patient confidentiality " * 4
    dense = [hit(1, 0.9, same), Hit("id2", 0.85, {"text": same, "content_hash": "other"}), hit(3, 0.8)]
    result = retrieve(FakeStore(dense, []))
    assert [c.id for c in result.chunks] == ["id1", "id3"]


# ----------------------------------------------------------------- query + prompt
def test_follow_up_questions_borrow_the_previous_topic():
    history = [("user", "What causes nephrotic syndrome?"), ("assistant", "…")]
    assert "nephrotic syndrome" in build_retrieval_query(history, "How is it treated?")
    assert build_retrieval_query(history, "Describe the pathophysiology of diabetic ketoacidosis in detail") == (
        "Describe the pathophysiology of diabetic ketoacidosis in detail"
    )
    assert build_retrieval_query([], "Why?") == "Why?"


def test_prompt_labels_sources_and_strips_old_citations_from_history():
    cand = type("C", (), {"payload": {"text": "Aspirin is contraindicated in children.", "chapter": "Chapter 9", "section_heading": "NSAIDs", "page_start": 10, "page_end": 11}})()
    assert "[S1] Chapter 9 | NSAIDs | PDF pp. 10-11\nAspirin" in format_context([cand])
    msgs = build_messages("Is aspirin safe for kids?", [cand], [("user", "q"), ("assistant", "A is bad [S2][S3, S4].")])
    assert msgs[0].content == SYSTEM_PROMPT and msgs[-1].role == "user"
    assert "[S" not in msgs[2].content and "Question: Is aspirin safe for kids?" in msgs[-1].content


def test_prompt_budget_drops_old_history_then_low_ranked_chunks():
    def chunk(n):
        return type("C", (), {"payload": {"text": ("word " * 700), "chapter": f"C{n}", "page_start": n, "page_end": n}})()

    chunks = [chunk(i) for i in range(1, 6)]
    history = [("user", "q" * 500), ("assistant", "a" * 500), ("user", "q2" * 250), ("assistant", "a2" * 250)]
    big = build_messages("Q?", chunks, history, max_input_tokens=None)
    assert len(big) == 6  # system + 4 history + user
    fit = build_messages("Q?", chunks, history, max_input_tokens=2500)
    total = sum(estimate_tokens(m.content) for m in fit)
    assert total <= 2500 or "[S3]" in fit[-1].content  # budget met, or floor of MIN_CONTEXT_CHUNKS reached
    assert len(fit) < len(big) or "[S5]" not in fit[-1].content
    tiny = build_messages("Q?", chunks, history, max_input_tokens=10)
    assert "[S3]" in tiny[-1].content and "[S4]" not in tiny[-1].content and len(tiny) == 2  # floor, no history


def test_rate_limiter_window():
    t = [0.0]
    lim = SlidingWindowLimiter(2, 60.0, clock=lambda: t[0])
    assert lim.check("a")[0] and lim.check("a")[0]
    ok, retry = lim.check("a")
    assert not ok and 59 <= retry <= 60
    assert lim.check("b")[0]
    t[0] = 61.0
    assert lim.check("a")[0]


def test_fastembed_backend_normalises_and_prefixes_queries(monkeypatch):
    import sys
    import types

    import numpy as np

    seen = []

    class FakeTextEmbedding:
        def __init__(self, name, cache_dir=None):
            self.name = name

        def embed(self, texts, batch_size=32):
            for t in texts:
                seen.append(t)
                yield np.array([3.0, 4.0, 0.0], dtype="float32")

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    from app.core.embeddings import BGE_QUERY_PREFIX, make_embedder

    settings = types.SimpleNamespace(
        embedding_backend="fastembed", embedding_model="BAAI/bge-small-en-v1.5", embedding_dim=3,
        embedding_batch_size=8, fastembed_cache_dir="",
    )
    emb = make_embedder(settings)
    vec = emb.embed_query("heart")
    assert [round(x, 2) for x in vec] == [0.6, 0.8, 0.0]
    assert seen[-1] == BGE_QUERY_PREFIX + "heart" and emb.dimension == 3 and emb.is_loaded
    try:
        make_embedder(types.SimpleNamespace(embedding_backend="nope"))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend must fail")
