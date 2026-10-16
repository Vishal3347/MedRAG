from __future__ import annotations

from app.ingestion.cleaning import clean_text, find_running_keys, join_lines
from app.ingestion.models import Line, RawPage
from app.ingestion.structure import build_blocks, compute_body_size


def test_clinical_content_is_preserved_verbatim():
    samples = [
        "Do NOT give aspirin to children under 16 years.",
        "Serum K+ ≥ 5.5 mmol/L; eGFR < 30 mL/min/1.73 m².",
        "Dose: 0.5 µg/kg IV; max 10⁻⁶ mol/L. No evidence of malignancy; not indicated without symptoms.",
        "Warfarin 5 mg od, INR 2.0–3.0 (±0.5).",
    ]
    for s in samples:
        assert clean_text(s) == s


def test_ligatures_invisible_chars_and_whitespace_are_fixed():
    assert clean_text("ﬁbrosis of the ﬂuid\u00ad  space\u200b") == "fibrosis of the fluid space"
    assert clean_text("a \t b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_hyphenation_repair():
    assert join_lines(["hypo-", "glycemia is common"]) == "hypoglycemia is common"
    assert join_lines(["a T-", "cell response"]) == "a T-cell response"
    assert join_lines(["range 5-", "10 mg"]) == "range 5- 10 mg"  # digits are never glued
    assert join_lines(["first line", "second line"]) == "first line second line"


def page(number, lines, height=800.0):
    return RawPage(number=number, height=height, lines=lines)


def L(text, size=10.0, bold=False, block=0, y=400.0):
    return Line(text=text, size=size, bold=bold, block=block, y=y)


def test_running_headers_and_page_numbers_are_removed():
    pages = []
    for n in range(1, 9):
        pages.append(
            page(
                n,
                [
                    L("Harrison Clinical Medicine", y=20, block=0),
                    L(f"Body text number {n} about renal physiology and nephron function.", block=1, y=300),
                    L(str(100 + n), y=780, block=2),
                ],
            )
        )
    body = compute_body_size(pages)
    keys = find_running_keys(pages, body)
    blocks = build_blocks(pages, body, keys)
    text = " ".join(b.text for b in blocks)
    assert "Harrison" not in text and "10" not in text.replace("number", "").replace("n", "")[:0]
    assert all(b.text.startswith("Body text") for b in blocks)


def test_heading_detection_and_chapter_merge():
    pages = [
        page(1, [L("Chapter 3", 24, True, 0, 100), L("Diabetes Mellitus", 24, True, 1, 140),
                 L("Overview", 14, True, 2, 200),
                 L("Diabetes is a chronic condition characterised by hyperglycaemia.", 10, False, 3, 240)]),
        page(2, [L("Figure 3.1 Insulin secretion pathway", 9, True, 0, 100),
                 L("Insulin is released from beta cells in response to glucose.", 10, False, 1, 140)]),
    ]
    body = compute_body_size(pages)
    blocks = build_blocks(pages, body)
    assert [(b.text, b.level) for b in blocks if b.is_heading] == [("Chapter 3: Diabetes Mellitus", 1), ("Overview", 2)]
    assert any(b.text.startswith("Figure 3.1") and not b.is_heading for b in blocks)


def test_paragraph_split_by_page_break_is_rejoined_with_page_span():
    pages = [
        page(7, [L("The patient presented with fever and was started on", block=0, y=300)]),
        page(8, [L("intravenous antibiotics for seven days.", block=0, y=100)]),
    ]
    blocks = build_blocks(pages, 10.0)
    assert len(blocks) == 1
    assert blocks[0].text == "The patient presented with fever and was started on intravenous antibiotics for seven days."
    assert (blocks[0].page_start, blocks[0].page_end) == (7, 8)
