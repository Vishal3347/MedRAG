from __future__ import annotations

import re

from app.ingestion.chunking import Chunker, split_sentences
from app.ingestion.models import Block
from app.ingestion.tokens import approx_token_count


def para(tag: str, sentences: int = 8) -> str:
    return " ".join(
        f"Sentence {tag}{i} reports that hypertension treatment lowered systolic pressure by 12 mmHg in adults."
        for i in range(sentences)
    )


def H(text, level, page=1):
    return Block(text, page, page, True, level)


def P(text, page=1):
    return Block(text, page, page)


def make(target=300, overlap=50, **kw):
    return Chunker(document_id="d1", document_title="Book", target_tokens=target, overlap_tokens=overlap, **kw)


def test_headings_are_attached_as_metadata_and_inline_text():
    blocks = [H("Chapter 1: Cardiology", 1, 4), H("Hypertension", 2, 4), P(para("a", 3), page=4)]
    chunks = make().chunk(blocks)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chapter == "Chapter 1: Cardiology" and c.section_heading == "Hypertension"
    assert "## Hypertension" in c.text and c.page_start == 4
    assert c.embed_text.startswith("Chapter 1: Cardiology > Hypertension\n")


def test_chunks_respect_target_size_and_overlap_between_neighbours():
    blocks = [H("Chapter 2", 1)] + [P(para(f"p{i}_", 5), page=i + 1) for i in range(12)]
    chunks = make(target=300, overlap=60).chunk(blocks)
    assert len(chunks) > 3
    assert all(c.token_count <= 300 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail_sentence = split_sentences(prev.text)[-1]
        assert tail_sentence in nxt.text, "consecutive chunks should share trailing context"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_crosses_a_chapter_boundary():
    blocks = [H("Chapter 1", 1), P(para("x", 4)), H("Chapter 2", 1, 9), P(para("y", 4), 9)]
    chunks = make().chunk(blocks)
    assert [c.chapter for c in chunks] == ["Chapter 1", "Chapter 2"]
    assert "Sentence x" in chunks[0].text and "Sentence y" not in chunks[0].text


def test_small_sections_are_merged_but_large_ones_start_a_new_chunk():
    small = [H("Chapter 1", 1), H("A", 2), P("Tiny paragraph about A."), H("B", 2), P("Tiny paragraph about B.")]
    assert len(make().chunk(small)) == 1
    big = [H("Chapter 1", 1), H("A", 2), P(para("a", 8)), H("B", 2), P(para("b", 8))]
    chunks = make(target=600, overlap=50).chunk(big)
    assert [c.section_heading for c in chunks] == ["A", "B"]


def test_oversized_paragraph_is_split_on_sentences():
    chunks = make(target=120, overlap=20).chunk([H("Chapter 1", 1), P(para("z", 30))])
    assert len(chunks) > 2 and all(c.token_count <= 120 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "Sentence z0" in joined and "Sentence z29" in joined


def test_single_giant_sentence_is_hard_wrapped():
    giant = " ".join(["word"] * 500) + "."
    chunks = make(target=100, overlap=10).chunk([P(giant)])
    assert len(chunks) >= 5 and all(c.token_count <= 100 for c in chunks)


def test_page_spans_and_stable_hashes():
    blocks = [H("Chapter 1", 1), Block(para("m", 2), 5, 6)]
    first = make().chunk(blocks)
    second = make().chunk(blocks)
    assert (first[0].page_start, first[0].page_end) == (5, 6)
    assert [c.content_hash for c in first] == [c.content_hash for c in second]
    assert len(first[0].content_hash) == 32


def test_heading_stays_with_the_text_it_introduces_across_a_size_break():
    blocks = [H("Chapter 1", 1), P(para("a", 5)), H("Next topic", 2), P(para("b", 5))]
    chunks = make(target=200, overlap=0, section_break_tokens=10_000).chunk(blocks)
    holder = [c for c in chunks if "## Next topic" in c.text]
    assert holder and "Sentence b0" in holder[0].text.split("## Next topic")[1]


def test_split_sentences_respects_medical_abbreviations():
    text = "Give 5 mg/kg, e.g. in children. See Fig. 3 for details. Dr. Smith agrees."
    assert split_sentences(text) == ["Give 5 mg/kg, e.g. in children.", "See Fig. 3 for details.", "Dr. Smith agrees."]


def test_token_counter_is_monotonic_and_nonzero():
    assert approx_token_count("") == 0
    assert approx_token_count("hypoglycemia") > approx_token_count("the")
